"""Run actual static and temporally sparse observation blocks against the dense source."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_nonzero_observation import NonzeroObservationConv
from opencood.tools.train_int_silent_task import (
    digest, loader, prepare_stream, read, spatial_input, to_device, write_json)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    baseline = read(a.audit/'contract.json')
    assert read(a.audit/'report.json')['status']=='COMPLETED'
    assert baseline['plan']==digest(a.plan)
    assert baseline['frozen_source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    source,cfg,*_ = prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    plan = read(a.plan)
    rows = [r for r in plan['rows']['fit'] if r['scene'] in baseline['scenes']]
    trace = [json.loads(l) for l in (a.audit/'frames.jsonl').read_bytes().splitlines()]
    assert [r['frame'] for r in rows] == [r['frame'] for r in trace]
    for row in rows:
        assert row['split']=='train' and digest(Path(row['pcd']))==plan['members'][row['pcd']]
    _,stream = loader(rows,cfg,0,303)
    obs = source.core.observation
    dense = obs[0].conv2
    sparse = NonzeroObservationConv(dense).eval()
    # Copy before downstream batch normalization / in-place activations.
    inside = []
    hook = sparse.register_forward_hook(lambda m,inputs,out:inside.append(out.detach().clone()))
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        plan=baseline['plan'],source=baseline['frozen_source'],audit_frames=digest(a.audit/'frames.jsonl'),
        scenes=baseline['scenes'],layer='0.conv2',dense_atol=.0005,dense_rtol=.0001,scoring=False))
    with (a.output/'frames.jsonl').open('x',buffering=1) as file:
        for i,(sample,entry) in enumerate(zip(stream,trace)):
            assert sample['meta']['frame']==entry['frame']
            spatial = spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            obs[0].conv2 = dense
            gold = obs(spatial)
            obs[0].conv2 = sparse
            sparse.begin(False,False)
            static = obs(spatial)
            full = sparse.visits
            sparse.begin(True,entry['refresh'] is not None)
            temporal = obs(spatial)
            partial = sparse.visits
            assert len(inside)==2 and torch.equal(inside[0],inside[1]) and torch.equal(temporal,static)
            inside.clear()
            expected = entry['layers']['0.conv2']
            assert full['mac']==expected['static_mac'] and partial['mac']==expected['clean_reuse_mac']
            assert partial['sites']==expected['sites']-expected['clean_sites']
            err = (temporal-gold).abs()
            rec = dict(frame=entry['frame'],scene=entry['scene'],refresh=entry['refresh'],static=full,
                temporal=partial,exact_layer=True,exact_observation=True,dense_max_error=float(err.max()),
                dense_violations=int((err>(.0005+.0001*gold.abs())).sum()),
                cache_bytes=sum(v.numel()*v.element_size() for v in sparse.previous))
            file.write(json.dumps(rec,allow_nan=False)+'\n')
    assert i+1==len(rows)
    hook.remove()
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,optimizer_steps=0,detection_scores=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('long-base','plan','audit','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)

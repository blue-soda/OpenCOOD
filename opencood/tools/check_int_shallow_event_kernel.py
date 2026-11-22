"""Compare true generic MAC and signed accumulation on identical shallow membrane levels."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_shallow_pulse import ShallowPulseConv
from opencood.tools.train_int_silent_task import (
    digest, loader, prepare_stream, read, spatial_input, to_device, write_json)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    contract = read(a.exact/'contract.json')
    assert read(a.exact/'report.json')['status']=='COMPLETED' and contract['parent']['plan']==digest(a.plan)
    assert contract['parent']['frozen_source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    plan = read(a.plan)
    rows = [r for r in plan['rows']['fit'] if r['scene'] in contract['scenes']]
    trace = [json.loads(l) for l in (a.exact/'frames.jsonl').read_bytes().splitlines()]
    assert [r['frame'] for r in rows]==[r['frame'] for r in trace]
    for r in rows:
        assert r['split']=='train' and digest(Path(r['pcd']))==plan['members'][r['pcd']]
    source,cfg,*_ = prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    block = source.core.observation[0]
    paid = ShallowPulseConv(block.conv2,signed=False)
    signed = ShallowPulseConv(block.conv2)
    # Read the actual input of the running detector's targeted convolution.
    inputs = []
    hook = block.conv2.register_forward_pre_hook(lambda module,args:inputs.append(args[0].clone()))
    _,stream = loader(rows,cfg,0,303)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        arm=a.arm,plan=contract['parent']['plan'],source=contract['parent']['frozen_source'],
        scenes=contract['scenes'],exact=digest(a.exact/'frames.jsonl'),scoring=False,optimizer_steps=0))
    with (a.output/'frames.jsonl').open('x',buffering=1) as f:
        for i,(sample,ref) in enumerate(zip(stream,trace)):
            meta=sample['meta']
            assert meta['frame']==ref['frame']
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            block(spatial)
            assert len(inputs)==1
            x=inputs.pop()
            results={}
            for name,conv in [('signed',signed),('mac',paid)]:
                conv.begin(a.arm,ref['refresh'],meta,spatial.shape[2:],source.bounds,True)
                conv(x)
                results[name]=conv.record
            assert torch.equal(paid.states[a.arm][1],signed.states[a.arm][1])
            difference=(paid.states[a.arm][2]-signed.states[a.arm][2]).abs()
            rec=dict(frame=meta['frame'],scene=meta['scene'],refresh=ref['refresh'],arms=results,
                     matched_levels=True,max_interkernel_error=float(difference.max()))
            f.write(json.dumps(rec,allow_nan=False)+'\n')
    hook.remove()
    assert i+1==len(rows)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,labels=0,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('arm',choices=('instant','motion_unit','motion_amplitude'))
    for key in ('plan','exact','long-base','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

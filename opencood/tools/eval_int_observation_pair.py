"""Score static and temporally reused observation convolutions in the same frozen detector."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_nonzero_observation import NonzeroObservationConv
from opencood.tools.eval_int_address_cross import (
    AddressWriter, BranchFusion, CachedResidualPolicy, EdgeView, GroupedEdge, InspectMiddle,
    ObservedPolicy, SignedProjectionAccumulator, SilencePolicy, TaskDrive, build, compact_group_bucket,
    conv_costs, digest, loader, make_trace, prepare_stream, read, run_step, score, spatial_input,
    to_device, write_json)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    parent = read(a.cross/'contract.json')
    assert parent['group']=='original' and read(a.cross/'report.json')['status']=='COMPLETED'
    assert parent['weights']==digest(a.fit/'adapters.pt') and parent['plan']==digest(a.plan)
    assert parent['quota']==digest(a.quota) and parent['drive']==digest(a.drive/'report.json')
    assert parent['frozen_source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    measured = [json.loads(l) for l in (a.live/'frames.jsonl').read_bytes().splitlines()]
    baseline = [json.loads(l) for l in (a.cross/'frames.jsonl').read_bytes().splitlines()]
    plan = read(a.plan)
    rows = [r for r in plan['rows']['fit'] if r['scene'] in parent['scenes']]
    for r in rows:
        assert r['split']=='train'
        for key in ('pcd','label'):
            if r[key]:assert digest(Path(r[key]))==plan['members'][r[key]]
    source,cfg,_,_,_,_,q = prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    core = source.core
    fusion = BranchFusion(core.readout.layers[0].double_conv[0][0]).eval()
    core.readout.layers[0].double_conv[0][0] = fusion
    conv = NonzeroObservationConv(core.observation[0].conv2).eval()
    core.observation[0].conv2 = conv
    visits = []
    hook = conv.register_forward_hook(lambda m,inputs,out:visits.append(m.visits))
    edge = GroupedEdge(source.bounds,16384,read(a.long_base/'book_train/report.json')['levels'])
    rates = read(a.quota)['models']['age']['rates']
    coeff = read(a.drive/'report.json')['fit']['coefficients']
    weights = torch.load(a.fit/'adapters.pt',map_location='cpu')['amplitude']
    names = ('static','temporal')
    models,writers = {},{}
    for n in names:
        model = build(source,q,EdgeView(edge,'tile'))
        gate = SilencePolicy(edge.arms['tile'],'silent')
        exe = InspectMiddle(False,model.coarse_executor.receive_fine,gate)
        projector = model.coarse_projector
        def grant(count,p=projector):
            return int(count*rates[str(p.age)]+.5)
        writer = AddressWriter(TaskDrive(coeff,'continuous'),exe,grant,'amplitude',False)
        writer.adapter.copy_(weights)
        exe.drive = writer
        projector.event_linear = SignedProjectionAccumulator(projector,False)
        projector.event_observer = writer.observe_events
        model.coarse_executor = exe
        models[n],writers[n] = model,writer
    costs = conv_costs(core)
    reasons = make_trace(models['static'],rows)
    ds,stream = loader(rows,cfg,0,303)
    assert len(rows)==len(measured)==len(baseline)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        parent=parent,live_trace=digest(a.live/'frames.jsonl'),dense_trace=digest(a.cross/'frames.jsonl'),
        adapter='amplitude',training=False,scenes=parent['scenes']))
    factory = partial(CachedResidualPolicy,compiler=compact_group_bucket)
    labels = 0
    with (a.output/'frames.jsonl').open('x',buffering=1) as file:
        for i,(sample,reason,ref,base) in enumerate(zip(stream,reasons,measured,baseline)):
            meta = sample['meta']
            assert ref['frame']==base['frame']==meta['frame']==rows[i]['frame']
            assert ref['refresh']==base['refresh']==reason
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                states = dict.fromkeys(names)
                selectors = {n:ObservedPolicy(models[n],factory) for n in names}
            ego = to_device(sample['ego'],torch.device('cuda'))
            spatial = spatial_input(source,ego)
            edge.start(meta,spatial.shape[-2:],reason)
            gt = ds.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            labels += int(gt is not None)
            item = dict(frame=meta['frame'],scene=meta['scene'],refresh=reason,arms={})
            if gt is not None:item['points'] = base['points']
            first_pred = None
            for n in names:
                conv.begin(n=='temporal',n=='temporal' and reason is not None)
                pred,states[n],rec = run_step(models[n],writers[n],selectors[n],fusion,spatial,meta,
                    states[n],writers[n].charge,reason,'A',None,costs)
                assert len(visits)==1 and visits[0]==ref[n]
                rec['observation_mac'] = visits.pop()['mac']
                if gt is not None:rec.update(score(ds.post,ego,pred,gt,record_matches=True))
                if first_pred is None:first_pred = pred
                else:assert all(torch.equal(v,first_pred[k]) for k,v in pred.items())
                item['arms'][n] = rec
            x,y = item['arms'].values()
            assert {k:v for k,v in x.items() if k!='observation_mac'}=={k:v for k,v in y.items() if k!='observation_mac'}
            file.write(json.dumps(item,allow_nan=False)+'\n')
    hook.remove()
    assert i+1==len(rows)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,labels=labels,optimizer_steps=0))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    for key in ('long-base','plan','quota','drive','fit','cross','live','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)

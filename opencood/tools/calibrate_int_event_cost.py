"""Price channel quantization on legal shallow convolution routes without detection labels."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_shallow_anchor import AnchoredObservationConv
from opencood.models.sub_modules.int_shallow_pulse import receptive_counts
from opencood.tools.train_int_silent_task import digest,loader,prepare_stream,read,spatial_input,to_device,write_json


def allocate(cost,error):
    inventory=np.repeat(np.arange(3),[32,64,32])
    indices=np.arange(128)
    cap=error[:,1].sum()
    def choose(lam):
        objective=cost/cost[:,1].sum()+lam*error/cap
        _,slots=linear_sum_assignment(objective[:,inventory])
        return inventory[slots]
    lo,hi=0.,1.
    chosen=choose(lo)
    if error[indices,chosen].sum()>cap:
        while error[indices,choose(hi)].sum()>cap:hi*=2
        for _ in range(52):
            mid=(lo+hi)*.5
            if error[indices,choose(mid)].sum()>cap:lo=mid
            else:hi=mid
        chosen=choose(hi)
        lam=hi
    else:lam=0.
    return dict(scales=np.array([.125,.25,.5])[chosen].tolist(),lambda_normalized=lam,
                predicted_g=float(cost[indices,chosen].sum()),weighted_error=float(error[indices,chosen].sum()))


def routes(levels,codes,previous):
    change=codes-previous
    one=change.abs()==1
    _,du,dm=receptive_counts(levels,one,(change!=0)&~one)
    one=codes.abs()==1
    _,ru,rm=receptive_counts(levels,one,(codes!=0)&~one)
    one=levels.abs()==1
    _,au,am=receptive_counts(levels,one,(levels!=0)&~one)
    dirty=(du+dm)>0
    dc,rc,tc=au+2*am,ru+2*rm,du+2*dm
    fresh=dirty & (dc<=rc) & (dc<=tc)
    anchor=dirty & ~fresh & (rc<=tc)
    h,w=levels.shape[2:]
    masks=torch.stack((fresh,anchor,~(fresh|anchor))).reshape(3,h,w).short()
    pad=F.pad(masks,(1,1,1,1))
    return sum(pad[:,y:y+h,x:x+w] for y in range(3) for x in range(3))


@torch.no_grad()
def run(a):
    table=read(a.calibration)
    plan=read(a.plan)
    assert table['plan']==digest(a.plan) and table['trace']==digest(a.cross/'frames.jsonl')
    rows=[dict(r,label=None,supervised=False) for r in plan['rows']['fit'] if r['scene'] in table['scenes']]
    for r in rows:assert r['split']=='train' and digest(Path(r['pcd']))==plan['members'][r['pcd']]
    trace=[(r['frame'],r['refresh']) for r in map(json.loads,(a.cross/'frames.jsonl').read_bytes().splitlines())]
    assert [r['frame'] for r in rows]==[f for f,_ in trace]
    source,cfg,*_=prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    assert table['source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    source.eval().requires_grad_(False)
    _,stream=loader(rows,cfg,0,303)
    block=source.core.observation[0]
    actual=AnchoredObservationConv(block.conv2,signed_rebuild=True).eval()
    quanta=(.125,.25,.5)
    counts=torch.zeros((2,128,3),device=block.conv2.weight.device,dtype=torch.long)
    ordinary=0
    for sample,(frame,reason) in zip(stream,trace):
        meta=sample['meta']
        assert meta['frame']==frame and not meta['supervised']
        spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
        x=block.relu(block.bn1(block.conv1(spatial)))
        actual.begin('anchor_absolute',reason,meta,spatial.shape[2:],source.bounds,True)
        actual(x)
        if reason is not None:
            base=x
            previous=[torch.zeros_like(x) for _ in quanta]
            continue
        ordinary+=1
        levels=[(x/q).round() for q in quanta]
        codes=[(v*q-base)/q for v,q in zip(levels,quanta)]
        fanout=routes(levels[1],codes[1],previous[1])
        for j,q in enumerate(quanta):
            update=torch.zeros_like(counts[:,:,j])
            for route,values in zip(fanout,(levels[j],codes[j],codes[j]-previous[j])):
                nonzero=((values!=0)*route).sum((0,2,3))
                unit=((values.abs()==1)*route).sum((0,2,3))
                update[0]+=2*nonzero
                update[1]+=2*nonzero-unit
            counts[:,:,j]+=128*update
            if q==.25:
                rec=actual.record
                charge=sum(2*rec[k+'_mac']+rec.get(k+'_ac',0) for k in ('reference','delta','rebuild'))
                assert charge==128*int(update[1].sum())
        previous=codes
    assert ordinary==table['ordinary']
    cost=counts.cpu().numpy()/(ordinary*1e9)
    error=np.asarray(table['cost'])
    names=('generic','event')
    write_json(a.output/'report.json',dict(status='COMPLETED',code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        calibration=digest(a.calibration),scenes=table['scenes'],ordinary=ordinary,labels=0,optimizer_steps=0,
        route='uniform .25 signed minimum; use previous on stable outputs',unit='G per ordinary scan',
        costs={n:v.tolist() for n,v in zip(names,cost)},assignments={n:allocate(v,error) for n,v in zip(names,cost)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('long-base','plan','cross','calibration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

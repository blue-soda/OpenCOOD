"""Assign a fixed inventory of shallow activation quanta using unlabeled training scans."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from opencood.tools.train_int_silent_task import (
    digest, loader, prepare_stream, read, spatial_input, to_device, write_json)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    parent=read(a.cross/'contract.json')
    assert parent['group']=='original' and read(a.cross/'report.json')['status']=='COMPLETED'
    assert parent['plan']==digest(a.plan) and parent['frozen_source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    plan=read(a.plan)
    rows=[dict(r,label=None,supervised=False) for r in plan['rows']['fit'] if r['scene'] in parent['scenes']]
    for r in rows:
        assert r['split']=='train' and digest(Path(r['pcd']))==plan['members'][r['pcd']]
    trace=[(r['frame'],r['refresh']) for r in map(json.loads,(a.cross/'frames.jsonl').read_bytes().splitlines())]
    assert [r['frame'] for r in rows]==[f for f,_ in trace]
    source,cfg,*_=prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    _,stream=loader(rows,cfg,0,303)
    block=source.core.observation[0]
    gain=block.conv2.weight.double().square().sum((0,2,3))
    quanta=np.array([.125,.25,.5])
    inventory=np.repeat(np.arange(3),[32,64,32])
    error=gain.new_zeros(128,3)
    values=ordinary=0
    for sample,(frame,reason) in zip(stream,trace):
        assert sample['meta']['frame']==frame and not sample['meta']['supervised']
        if reason is not None:continue
        x=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
        x=block.relu(block.bn1(block.conv1(x)))
        for j,q in enumerate(quanta):
            err=x-(x/q).round()*q
            error[:,j]+=err.double().square().sum((0,2,3))
        ordinary+=1
        values+=x.shape[-2]*x.shape[-1]
    assert ordinary==sum(reason is None for _,reason in trace)==353
    cost=(error*gain[:,None]/values).cpu().numpy()
    channels,slots=linear_sum_assignment(cost[:,inventory])
    assert np.array_equal(channels,np.arange(128))
    selected=inventory[slots]
    shuffled=np.random.RandomState(303).permutation(selected)
    write_json(a.output/'report.json',dict(status='COMPLETED',code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        scenes=parent['scenes'],plan=parent['plan'],source=parent['frozen_source'],trace=digest(a.cross/'frames.jsonl'),
        scans=len(rows),ordinary=ordinary,values_per_channel=values,quanta=quanta.tolist(),
        gain=gain.cpu().tolist(),sse=error.cpu().tolist(),cost=cost.tolist(),
        assigned=quanta[selected].tolist(),shuffled=quanta[shuffled].tolist(),labels=0,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cross','plan','long-base','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

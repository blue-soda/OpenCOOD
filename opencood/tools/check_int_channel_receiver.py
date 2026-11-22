"""Factor static signed weights and temporal reuse on locked channel-grid observations."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_shallow_anchor import AnchoredObservationConv
from opencood.models.sub_modules.int_bounded_sender import BoundedObservationConv
from opencood.tools.train_int_silent_task import digest,loader,prepare_stream,read,spatial_input,to_device,write_json


@torch.no_grad()
def run(a):
    source,cfg,*_=prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    parent=read(a.cross/'contract.json')
    table=read(a.calibration)
    plan=read(a.plan)
    assert parent['plan']==table['plan']==digest(a.plan)
    assert parent['frozen_source']==table['source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    probe=a.rule in ('history_escape','current_escape','exact_escape','hold_only')
    if probe and parent['group']=='original':
        assert parent['scenes']==table['scenes']
    else:
        assert parent['group']=='additional' and not set(parent['scenes']).intersection(table['scenes'])
    rows=[dict(r,label=None,supervised=False) for r in plan['rows']['fit'] if r['scene'] in parent['scenes']]
    trace=[(r['frame'],r['refresh']) for r in map(json.loads,(a.cross/'frames.jsonl').read_bytes().splitlines())]
    assert [f for f,_ in trace]==[r['frame'] for r in rows]
    for r in rows:assert r['split']=='train' and digest(Path(r['pcd']))==plan['members'][r['pcd']]
    scales=table['assigned'] if a.rule=='assigned' else None
    if a.rule in ('generic','event'):
        allocation=read(a.cost_calibration)
        assert allocation['status']=='COMPLETED' and allocation['calibration']==digest(a.calibration)
        scales=allocation['assignments'][a.rule]['scales']
    mode='anchor_absolute' if scales else 'anchor_float'
    block=source.core.observation[0]
    specs={'legacy':(True,False,True),'static_mac':(False,False,False),'static_signed':(True,True,False),
           'temporal_mac':(False,False,True),'temporal_signed':(True,True,True)}
    if probe:
        convs={}
        for signed in (False,True):
            key='temporal_signed' if signed else 'temporal_mac'
            convs[key]=(AnchoredObservationConv(block.conv2,signed=signed,signed_rebuild=signed)
                        if a.rule=='exact_escape' else
                        BoundedObservationConv(block.conv2,history=a.rule!='current_escape',signed=signed,
                                               allow_unit=a.rule!='hold_only'))
    else:
        convs={n:AnchoredObservationConv(block.conv2,scales,*opts) for n,opts in specs.items()}
    first=next(iter(convs.values()))
    _,stream=loader(rows,cfg,0,303)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        calibration=digest(a.calibration),cost_calibration=digest(a.cost_calibration) if a.cost_calibration else None,
        parent=parent,rule=a.rule,scoring=False))
    with (a.output/'frames.jsonl').open('x',buffering=1) as f:
        for sample,(frame,reason) in zip(stream,trace):
            meta=sample['meta']
            assert frame==meta['frame'] and not meta['supervised']
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            x=block.relu(block.bn1(block.conv1(spatial)))
            results={}
            for n,conv in convs.items():
                conv.begin(mode,reason,meta,spatial.shape[2:],source.bounds,True)
                conv(x)
                results[n]=conv.record
                assert torch.equal(first.anchor_state[2],conv.anchor_state[2])
            f.write(json.dumps(dict(frame=frame,scene=meta['scene'],refresh=reason,arms=results),allow_nan=False)+'\n')
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=len(rows),optimizer_steps=0,labels=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('rule',choices=('assigned','float','generic','event','history_escape','current_escape','exact_escape','hold_only'))
    p.add_argument('--cost-calibration',type=Path)
    for k in ('long-base','plan','cross','calibration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

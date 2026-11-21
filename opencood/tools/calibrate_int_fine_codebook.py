"""Fit four scalar representatives on admitted training-only fine messages."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector,training_rows
from opencood.tools.check_int_deep_spike_stream import build_source,read
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import local_request,trace_masks


def fit(hist):
    x=np.arange(len(hist))
    hist=np.asarray(hist)
    table=np.array([1,2,4,8])
    for iteration in range(128):
        bins=np.abs(x[:,None]-table).argmin(axis=1)
        next_table=table.copy()
        for k in range(4):
            w=hist*(bins==k)
            if w.sum():next_table[k]=int(np.rint((x@w)/w.sum()))
        if np.array_equal(table,next_table):
            return table.tolist(),iteration
        table=next_table
    raise RuntimeError('Codebook fit did not converge')


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    plan=read(a.inputs/'plan.json')
    rows,traces,starts=training_rows(a.inputs,a.scans,plan)
    assert digest(a.prior)=='033356e8202a519b63db092e35f9f77fa232c31ba6b2abc8346a0f873412db76'
    old=[json.loads(l) for l in a.prior.read_bytes().splitlines()]
    cap=min(t['arms']['deferred']['fine']['addresses'] for t in old if t['refresh'] is None)//2
    cfg=load_yaml(str(a.inputs/'config.yaml'))
    cfg['data_augment']=[]
    source=build_source(a.inputs,cfg)
    q,_,_=load_quantum(a.inputs/'quantum.json',plan,512)
    model=detector(source,q,'carry',.25)
    edge=FineCodebook(source.bounds,cap)
    def after_select(fine,active):
        edge.active=active
        return edge(fine)
    model.fine_to_coarse=lambda fine:edge(fine) if edge.phase=='refresh' else fine
    model.coarse_executor=FineInnovationUpdate(False,receive_fine=after_select)
    model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
    _,stream=loader(rows,cfg,0,303)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               cap=cap,source=digest(a.prior),train=digest(a.scans/'train_plan.json'),params=sum(p.numel() for p in source.parameters()),
               optimizer_steps=0,labels_used=0))
    total=np.zeros(1,dtype=np.int64)
    n=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,ref in zip(stream,traces,old):
            meta=sample['meta']
            assert not meta['supervised'] and meta['frame']==trace['frame']==ref['frame']
            if meta['frame'] in starts:state=None
            reason=trace['refresh']
            assert model.refresh_reason(state,meta)==reason
            model.coarse_projector.begin(state,reason)
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            edge.begin(state,meta,spatial.shape[-2:],reason)
            _,state,_=model.step_spatial(spatial,state,meta,local_request(trace_masks(trace,spatial.device)))
            edge.commit(state,False)
            entry=dict(frame=meta['frame'],scene=meta['scene'],refresh=reason)
            if reason is None:
                h=edge.ledger['magnitudes']
                assert sum(h)==cap and h[0]==0
                if len(h)>len(total):total=np.pad(total,(0,len(h)-len(total)))
                total[:len(h)]+=h
                entry['hist']=h
                model.coarse_executor.inputs=None
            log.write(json.dumps(entry)+'\n')
            n+=1
    assert n==len(rows)==len(old)==256
    table,iters=fit(total)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=n,cap=cap,levels=table,iterations=iters,hist=total.tolist(),
               scenes=sorted({r['scene'] for r in rows}),optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('inputs','scans','prior','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)


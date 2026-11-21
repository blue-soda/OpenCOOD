"""Matched closed-loop BEV receiver transport, without labels or optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fine_registration import ReceiverGuide
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.eval_int_codebook_fraction import FullAmplitude
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_fraction_history import HistoryExperiment,numeric_mask


class ErrorAudit:
    def __call__(self,value):
        if self.phase=='refresh':
            self.errors = {}
            return super().__call__(value)
        self.target = (value/self.quantum).round().int().flatten(2)[0]
        output = super().__call__(value)
        delta = (self.target-self.levels.flatten(2)[0])[:,self.read.flatten()]
        self.errors.update(after=int(delta.abs().sum(dtype=torch.long)), cells=delta.numel())
        self.target = None
        return output

    def reference_levels(self,receiver,sites,area):
        prior = super().reference_levels(receiver,sites,area)
        self.errors = dict(before=int((self.target[:,sites]-prior).abs().sum(dtype=torch.long)))
        return prior


class AuditBook(ErrorAudit,FineCodebook):
    pass


class AuditAmplitude(ErrorAudit,FullAmplitude):
    pass


@torch.no_grad()
def run(a):
    source,cfg,_,_,_,_,q = prepare_stream(a.inputs,a.scans)
    book = read(a.book)
    plan = read(a.training/'plan.json')
    report = read(a.training/'report.json')
    assert report['plan']==digest(a.training/'plan.json') and report['frames']==digest(a.training/'frames.jsonl')
    assert digest(a.book)==plan['book'] and book['cap']==428149
    cap=(book['cap']//2)*a.eighths//8
    exp=HistoryExperiment(source,q,book)
    model,_=exp.models['carry']
    rows=[dict(r,supervised=False,label=None) for r in plan['rows']]
    assert all(digest(Path(r['pcd']))==plan['members'][r['pcd']] for r in rows)
    traces=[json.loads(s) for s in (a.training/'frames.jsonl').read_bytes().splitlines()]
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for name in (('book',) if a.direction_controls else ('book','amplitude')):
            strategies=('shared','opposite') if a.direction_controls else ('ego','local')
            for strategy in strategies:
                guided=strategy!='ego'
                edge=AuditBook(source.bounds,cap,book['levels']) if name=='book' else AuditAmplitude(source.bounds,'amplitude',cap)
                guide=ReceiverGuide(strategy) if guided else None
                if guide:
                    edge.transport=guide.propagate
                    hook=source.core.observation.register_forward_hook(guide.observe)
                model.fine_to_coarse=edge
                exp.models['carry']=model,edge
                _,stream=loader(rows,cfg,0,303)
                for i,(sample,trace) in enumerate(zip(stream,traces)):
                    meta=sample['meta']
                    assert not meta['supervised'] and 'label_dict' not in sample['ego']
                    if i==0 or rows[i-1]['segment']!=meta['segment']:
                        state=None
                    spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
                    _,state,stats=exp.step('carry',spatial,meta,state,trace,numeric_mask(trace))
                    if guide:
                        guide.save(state)
                    row=dict(frame=meta['frame'],scene=meta['scene'],name=name,guided=guided,age=state['since_refresh'],
                             task=hashlib.sha256(state['value'].contiguous().cpu().numpy().tobytes()).hexdigest(),**stats)
                    if a.direction_controls:
                        row['strategy']=strategy
                    if trace['refresh'] is None:
                        row.update(demand=edge.ledger['demand_addresses'],**edge.errors)
                        if guide:row.update(guide.cost)
                    log.write(json.dumps(row)+'\n')
                assert i+1==len(rows)==211
                if guide:hook.remove()
    write_json(a.output/'report.json',dict(status='COMPLETED',optimizer_steps=0,labels=0,scans=211,cap=cap,
        train=report['plan'],trace=report['frames'],book=digest(a.book),
        code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','training','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--eighths',type=int,choices=(7,8),required=True)
    p.add_argument('--direction-controls',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)


"""Compare receiver-error feedback to adjacent-target differencing at a fixed wire budget."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_fraction_history import HistoryExperiment,numeric_mask


class CapturedCodebook(FineCodebook):
    def deliver(self,receiver,ids,change):
        self.before=receiver.clone()
        self.ids=ids
        self.sent,extra=super().deliver(receiver,ids,change)
        return self.sent,extra


class FeedbackAudit(CapturedCodebook):
    def __init__(self,bounds,book):
        super().__init__(bounds,book['cap'],book['levels'])
        self.adjacent=CapturedCodebook(bounds,book['cap'],book['levels'])

    def __call__(self,value):
        prior=self.previous
        output=super().__call__(value)
        if self.phase=='refresh':
            self.other_receiver=self.levels.clone()
            self.audit=None
            return output
        # Give the alternative sender the previous uncompressed target, but
        # accumulate its emitted symbols into its own actual receiver.
        target=(value/self.quantum).round().int()
        previous_target=(prior['value']/self.quantum).round().int()
        alt=self.adjacent
        alt.begin(dict(fine_levels=previous_target,pose=prior['pose']),dict(frame=self.frame_number,pose=self.pose),self.shape,None)
        alt.active=self.active
        alt(value)
        assert torch.equal(self.read,alt.read)
        assert self.ledger['addresses']==alt.ledger['addresses']==self.cap
        bounds=block_lidar_range(self.bounds,self.shape,value.shape[-2:],2)
        other=warp_bev(self.other_receiver.float(),prior['pose'],self.pose,bounds)[0].int().contiguous()
        other.view(-1).index_add_(0,alt.ids,alt.sent)
        self.other_receiver=other
        common=self.before.clone().view_as(target)
        common.view(-1).index_add_(0,alt.ids,alt.sent)
        def error(levels):
            delta=(target-levels.view_as(target))[:,:,self.read[0,0]].long()
            return dict(l1=int(delta.abs().sum()),square=int(delta.square().sum()),cells=delta.numel())
        admitted=torch.zeros(target.numel(),dtype=torch.bool,device=target.device)
        admitted[self.ids]=True
        self.audit=dict(feedback=error(self.levels),adjacent=error(other),adjacent_one_step=error(common),
                        before=error(self.before),shared_addresses=int(admitted[alt.ids].sum()),addresses=self.cap,
                        bytes=self.ledger['bytes'],adjacent_bytes=alt.ledger['bytes'],units=self.ledger['units'],
                        adjacent_units=alt.ledger['units'])
        return output


@torch.no_grad()
def run(a):
    source,cfg,_,_,_,_,q=prepare_stream(a.inputs,a.scans)
    book=read(a.book)
    assert book['cap']==428149
    book=dict(book,cap=book['cap']//a.budget_divisor)
    exp=HistoryExperiment(source,q,book)
    model,_=exp.models['carry']
    edge=FeedbackAudit(source.bounds,book)
    model.fine_to_coarse=edge
    exp.models['carry']=(model,edge)
    plan=read(a.training/'plan.json')
    assert digest(a.training/'plan.json')==read(a.training/'report.json')['plan']
    assert digest(a.training/'frames.jsonl')==read(a.training/'report.json')['frames']
    assert digest(a.book)==plan['book']
    rows=[dict(r,label=None,supervised=False) for r in plan['rows']]
    assert all(digest(Path(r['pcd']))==plan['members'][r['pcd']] for r in rows)
    traces=[json.loads(line) for line in (a.training/'frames.jsonl').read_bytes().splitlines()]
    _,stream=loader(rows,cfg,0,303)
    ordinary=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace) in enumerate(zip(stream,traces)):
            meta=sample['meta']
            assert not meta['supervised'] and 'label_dict' not in sample['ego']
            if i==0 or rows[i-1]['segment']!=meta['segment']:state=None
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            _,state,_=exp.step('carry',spatial,meta,state,trace,numeric_mask(trace))
            row=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'],age=state['since_refresh'])
            if edge.audit is not None:
                row.update(edge.audit)
                ordinary+=1
            log.write(json.dumps(row)+'\n')
    assert i+1==len(rows)==211 and ordinary==165
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=len(rows),ordinary=ordinary,
               code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),labels=0,optimizer_steps=0,
               train=digest(a.training/'plan.json'),book=digest(a.book),cap=book['cap'],budget_divisor=a.budget_divisor))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','training','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--budget-divisor',type=int,choices=(1,2,4),default=1)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)


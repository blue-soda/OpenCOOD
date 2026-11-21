"""Factorial fixed-unit-budget intervention on middle feature decay and task carry."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import allocate, score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


class CappedProjection(MiddleDecayProjection):
    def __init__(self, block, predictor, retain, cap):
        super().__init__(block,predictor,'amplitude')
        self.retain=retain
        self.cap=cap

    def __call__(self, value, written):
        if not self.age:
            sent=(value/self.quantum).round()*self.quantum
            self.demand=0
            return sent,sent
        before=self.aligned[:,:value.shape[1]]
        target=value/self.quantum
        # Standard quantizer uses absolute levels, including its tie convention.
        delta=target.round()-before if self.predictor=='zero' else (target-before.float()/self.divisors[self.den]).round()
        ids,amps,self.demand=allocate(delta,self.cap,self.frame)
        self.events=torch.zeros_like(before,memory_format=torch.contiguous_format)
        self.events.view(-1)[ids]=amps.long()
        self.numerator=before+self.events*self.den
        self.maintenance=None
        sent=(self.numerator.float()/self.divisors[self.den])*.25
        return (value if self.retain else sent),sent


@torch.no_grad()
def run(inputs,scans,prior,out):
    source,cfg,rows,traces,starts,members,q=prepare_stream(inputs,scans)
    contract=read(prior/'contract.json')
    assert members==contract['members'] and contract['budget']==101767
    old=[json.loads(line) for line in (prior/'frames.jsonl').read_text().splitlines()]
    models={}
    for predict in ('zero','phase'):
        for store in ('carry','drop'):
            name=predict+'_'+store
            m=detector(source,q,'carry',.25)
            m.coarse_interface=m.coarse_projector=CappedProjection(source.core.deblocks[1],predict,store=='carry',contract['budget'])
            models[name]=m
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        prior=digest(prior/'frames.jsonl'),prior_contract=digest(prior/'contract.json'),members=members,
        weights={k:digest(inputs/k) for k in ('parent.pth','protected_light_final.pth','readout.pth','quantum.json')},
        params=sum(p.numel() for p in source.parameters()),cap=contract['budget'],quantum=.25,seed=303,
        trace=digest(inputs/'traces.json'),device=torch.cuda.get_device_name(),torch=torch.__version__))
    data,stream=loader(rows,cfg,0,303)
    n=0
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,ref in zip(stream,traces,old):
            meta=sample['meta']
            assert meta['frame']==trace['frame']==ref['frame']
            if meta['frame'] in starts:states=dict.fromkeys(models)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            request=local_request(trace_masks(trace,spatial.device))
            row=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'],arms={})
            for name,model in models.items():
                receiver=model.coarse_projector
                reason=model.refresh_reason(states[name],meta)
                assert reason==trace['refresh']
                receiver.begin(states[name],reason)
                receiver.frame=int(meta['frame'])
                (pred,state,info),core=measured_step(model,spatial,states[name],meta,request)
                states[name]=state
                m=receiver.ledger
                entry=dict(middle=m,deep=model.deep_projector.ledger,core=core,demand=receiver.demand)
                if reason is None:
                    assert m['units']==min(receiver.cap,receiver.demand)
                    if name.startswith('zero'):
                        before=ref['arms']['carry' if name=='zero_carry' else 'drop_all']
                        assert [m['units'],m['nonzero'],receiver.demand]==[before['units'],before['addresses'],before['demand']]
                if gt is not None:entry.update(score(data.post,ego,pred,gt))
                row['arms'][name]=entry
            log.write(json.dumps(row)+'\n');n+=1
    assert n==len(rows)==len(old)
    write_json(out/'report.json',dict(status='COMPLETED',scans=n,reference_counts_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('inputs','scans','prior','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.scans,args.prior,args.output)

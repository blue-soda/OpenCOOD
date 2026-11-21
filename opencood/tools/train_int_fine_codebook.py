"""Matched short distillation of ordinary fine-to-middle pointwise projections."""
import argparse
from pathlib import Path
import json
import subprocess

import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, supports
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector,training_rows
from opencood.tools.check_int_deep_spike_stream import build_source,read
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import local_request,trace_masks


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    plan=read(a.inputs/'plan.json')
    rows,traces,starts=training_rows(a.inputs,a.scans,plan)
    cfg=load_yaml(str(a.inputs/'config.yaml'))
    cfg['data_augment']=[]
    source=build_source(a.inputs,cfg)
    q,_,_=load_quantum(a.inputs/'quantum.json',plan,512)
    model=detector(source,q,'carry',.25)
    model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
    book=read(a.book)
    assert book['status']=='COMPLETED' and book['cap']==428149 and set(book['scenes'])=={r['scene'] for r in rows}
    cap=book['cap']
    senders=dict(power=FineCodebook(source.bounds,cap,[1,2,4,8]),
                 calibrated=FineCodebook(source.bounds,cap,book['levels']),amplitude=QuotaFineEdge(source.bounds,'amplitude',cap))
    params,opts,last={},{},{}
    channels=source.core.coarse_light[2].out_channels
    for n in senders:
        params[n]=(torch.ones(channels,device='cuda',requires_grad=True),torch.zeros(channels,device='cuda',requires_grad=True))
        opts[n]=torch.optim.Adam(params[n],lr=.001)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               book=digest(a.book),train=digest(a.scans/'train_plan.json'),params=sum(p.numel() for p in source.parameters()),
               trainable_per_arm=2*channels,lr=.001,seed=303,labels=0,objective='MSE to frozen ordinary continuous observation'))
    def finish(name,fine):
        record=dict(value=fine,pose=meta['pose'])
        senders[name].commit(record,False)
        last[name]=record
    def teach(fine,active):
        read_mask=supports(active)[2][0,0]
        def observation(v):
            return source.core.coarse_light(v)[0,:,read_mask]
        teacher=observation(fine)
        for name,edge in senders.items():
            edge.active=active
            x=observation(edge(fine))
            gain,offset=params[name]
            with torch.enable_grad():
                error=gain[:,None]*x+offset[:,None]-teacher
                loss=error.square().mean()
                opts[name].zero_grad()
                loss.backward()
                opts[name].step()
            record[name]=dict(loss=float(loss),unadapted=float((x-teacher).square().mean()),addresses=edge.ledger['addresses'])
            finish(name,fine)
        return fine
    model.coarse_executor=FineInnovationUpdate(False,receive_fine=teach)
    _,stream=loader(rows,cfg,0,303)
    n=steps=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace in zip(stream,traces):
            meta=sample['meta']
            assert not meta['supervised'] and meta['frame']==trace['frame']
            if meta['frame'] in starts:
                state=None
                last.clear()
            reason=trace['refresh']
            assert model.refresh_reason(state,meta)==reason
            spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            model.coarse_projector.begin(state,reason)
            for name,edge in senders.items():edge.begin(last.get(name),meta,spatial.shape[-2:],reason)
            record=dict(frame=meta['frame'],scene=meta['scene'],refresh=reason)
            _,state,_=model.step_spatial(spatial,state,meta,local_request(trace_masks(trace,spatial.device)))
            if reason is None:
                assert all(record[name]['addresses']==cap for name in senders)
                model.coarse_executor.inputs=None
                steps+=1
            else:
                for name,edge in senders.items():
                    edge(state['value'])
                    finish(name,state['value'])
            log.write(json.dumps(record)+'\n')
            n+=1
    assert n==256 and steps==200
    arms={name:dict(gain=g.tolist(),offset=b.tolist()) for name,(g,b) in params.items()}
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=n,steps=steps,arms=arms,book=digest(a.book)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)


"""Paired causal replay of eager and on-demand fine-to-middle messages."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate,conv_costs
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_fine_demand import DemandFineEdge
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.calibrate_int_middle_budget import training_rows
from opencood.tools.check_int_deep_spike_stream import build_source,prepare_stream,read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import local_request,trace_masks


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if a.training:
        plan=read(a.inputs/'plan.json')
        rows,traces,starts=training_rows(a.inputs,a.scans,plan)
        cfg=load_yaml(str(a.inputs/'config.yaml'))
        cfg['data_augment']=[]
        source=build_source(a.inputs,cfg)
        q,_,_=load_quantum(a.inputs/'quantum.json',plan,512)
    else:
        source,cfg,rows,traces,starts,_,q=prepare_stream(a.inputs,a.scans)
    old=[json.loads(line) for line in a.prior.read_bytes().splitlines()]
    models={}
    for name in ('eager','deferred'):
        sender=DemandFineEdge(source.bounds,name=='deferred')
        model=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
              deep_projector=DeepSpikeProjection(source.core.deblocks[2],q,'amplitude'),fine_to_coarse=sender.input_to_middle)
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
        model.coarse_executor=FineInnovationUpdate(False,receive_fine=sender.receive_jobs)
        models[name]=(model,sender)
    data,stream=loader(rows,cfg,0,303)
    costs=conv_costs(source.core)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               params=sum(p.numel() for p in source.parameters()),training_split=a.training,optimizer_steps=0,
               q=.25,reference=digest(a.prior),trace=digest(a.inputs/'traces.json'),device=torch.cuda.get_device_name(),
               weights={k:digest(a.inputs/k) for k in ('parent.pth','protected_light_final.pth','readout.pth','quantum.json')}))
    n=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,check in zip(stream,traces,old):
            meta=sample['meta']
            frame,reason=meta['frame'],trace['refresh']
            assert frame==trace['frame']==check['frame'] and reason==check['refresh']
            if frame in starts:
                states=dict.fromkeys(models)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            req=local_request(trace_masks(trace,spatial.device))
            row=dict(frame=frame,scene=meta['scene'],refresh=reason,arms={})
            for name,(model,sender) in models.items():
                prior=states[name]
                assert model.refresh_reason(prior,meta)==reason
                model.coarse_projector.begin(prior,reason)
                sender.begin(prior,meta,spatial.shape[-2:],reason)
                (pred,state,info),core=measured_step(model,spatial,prior,meta,req)
                sender.finish(state)
                exe=model.coarse_executor
                if reason is None:
                    core+=sum(costs[k]*v for k,v in exe.counts.items())
                    exe.inputs=None
                row['arms'][name]=dict(fine=sender.ledger,core=core)
                if name=='eager':
                    reference=pred
                    if not a.training:
                        archived=check['arms']['ordinary']
                        assert core==archived['core'] and sender.ledger==archived['fine']
                        if meta['supervised']:
                            gt=data.post.generate_gt_bbx(dict(ego=ego))
                            result=json.loads(json.dumps(score(data.post,ego,pred,gt)))
                            assert all(v==archived[k] for k,v in result.items())
                            row.update(result)
                    elif reason is None:
                        assert sender.ledger['addresses']==check['fine_events'] and sender.ledger['units']==check['fine_units']
                else:
                    assert all(torch.equal(reference[k],pred[k]) for k in pred)
                    for k in ('value','coarse','deep','coarse_projection','deep_projection'):
                        assert torch.equal(state[k],states['eager'][k]), (frame,k)
                    if reason is None:
                        assert torch.equal(exe.skip,models['eager'][0].coarse_executor.skip)
                        assert all(torch.equal(x,y) for x,y in zip(state['edge_cache'],states['eager']['edge_cache']))
                    assert core==row['arms']['eager']['core']
                states[name]=state
            log.write(json.dumps(row)+'\n')
            n+=1
    assert n==len(rows)==len(old)==(256 if a.training else 272)
    write_json(a.output/'report.json',dict(status='COMPLETED',frames=n,equal_heads=True,equal_masks=True,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','scans','prior','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training',action='store_true',help='Unlabelled training-split inference only.')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)


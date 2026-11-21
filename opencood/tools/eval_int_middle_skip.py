"""True compact middle convolution with causal 8x8 tile holds on the six frozen clips."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate,conv_costs
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.tools.calibrate_int_middle_budget import detector
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import local_request,trace_masks


@torch.no_grad()
def run(inputs,scans,prior,out):
    source,cfg,rows,traces,starts,members,q=prepare_stream(inputs,scans)
    assert members==read(prior/'contract.json')['members']
    checks=[json.loads(l) for l in (prior/'frames.jsonl').read_text().splitlines()]
    per=conv_costs(source.core)
    models={}
    for name in ('zero_zero','zero_phase','phase_phase'):
        hold,send=name.split('_')
        model=detector(source,q,'carry',.25)
        model.coarse_executor=FineInnovationUpdate(hold=='phase')
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],send,'amplitude')
        models[name]=model
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               reference=digest(prior/'frames.jsonl'),members=members,cost_per_site=per,quantum=.25,seed=303,
               params=sum(p.numel() for p in source.parameters()),device=torch.cuda.get_device_name(),torch=torch.__version__,
               trace=digest(inputs/'traces.json'),weights={k:digest(inputs/k) for k in ('parent.pth','protected_light_final.pth','readout.pth','quantum.json')}))
    data,stream=loader(rows,cfg,0,303);n=0
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,check in zip(stream,traces,checks):
            meta=sample['meta'];frame=meta['frame'];reason=trace['refresh']
            assert frame==trace['frame']==check['frame']
            if frame in starts:states=dict.fromkeys(models)
            ego=to_device(sample['ego'],torch.device('cuda'));spatial=spatial_input(source,ego)
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            request=local_request(trace_masks(trace,spatial.device))
            row=dict(frame=frame,scene=meta['scene'],refresh=reason,arms={})
            reference_mask=None
            for name,model in models.items():
                receiver=model.coarse_projector
                previous=states[name]
                assert model.refresh_reason(previous,meta)==reason
                receiver.begin(previous,reason)
                (pred,state,info),other=measured_step(model,spatial,previous,meta,request)
                states[name]=state
                m=model.coarse_executor
                cost=0
                entry=dict(middle=receiver.ledger,deep=model.deep_projector.ledger)
                ref_core=check['arms']['phase']['core_float_flops']
                if reason is None:
                    sites=m.skip.numel()
                    assert other+sites*sum(per.values())==ref_core
                    cost=sum(per[k]*v for k,v in m.counts.items())
                    if reference_mask is None:reference_mask=m.skip
                    else:assert torch.equal(reference_mask,m.skip)
                    entry.update(support=m.counts,skipped=int(m.skip.sum()),gain=m.gain)
                    # Independently align previous integer levels and reconstruct messages on held sites.
                    shape=state['coarse'].shape
                    from opencood.models.sub_modules.int_block_cache import block_lidar_range
                    area=block_lidar_range(source.bounds,spatial.shape[-2:],shape[-2:],4)
                    # Call base coordinate router, without preparing the production receiver a second time.
                    from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
                    _,_,(idx,missing)=DeepPhaseProjection.align_task(receiver,previous['coarse'][:,:1],previous['pose'],meta['pose'],area)
                    old=previous['coarse_projection'][:,:shape[1]].flatten(2).index_select(2,idx).view(shape).masked_fill_(missing,0)
                    delta=state['coarse_projection'][:,:shape[1]]-old
                    units=delta.abs().sum(dtype=torch.float64)/receiver.den
                    assert float(units)==entry['middle']['units']
                    entry['held_units']=int(delta.masked_select(m.skip).abs().sum()/receiver.den)
                    if name=='phase_phase':
                        fine,old,valid,held=m.inputs
                        dense=model.core.coarse_update(model.core.coarse_light(fine),old,valid)
                        ref=torch.where(m.skip,held,dense)
                        entry['max_abs']=float((state['coarse']-ref).abs().max())
                        assert torch.allclose(state['coarse'],ref,atol=1e-4,rtol=1e-5),frame
                        assert torch.equal(state['coarse'].masked_select(m.skip),held.masked_select(m.skip))
                    m.inputs=None
                else:
                    assert other==ref_core
                entry.update(core=other+cost,middle_conv=cost,reference_core=ref_core)
                if gt is not None:entry.update(score(data.post,ego,pred,gt))
                row['arms'][name]=entry
            log.write(json.dumps(row)+'\n');n+=1
    assert n==len(rows)==272
    write_json(out/'report.json',dict(status='COMPLETED',scans=n,same_masks=True,same_state_dense_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','scans','prior','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run(a.inputs,a.scans,a.prior,a.output)

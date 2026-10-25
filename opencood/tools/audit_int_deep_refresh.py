"""Train-only dependency and stale-feature audit for periodic deep refresh."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_deep_dependencies import deep_dependencies
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter


def audit_deep(a,cfg,train,mode,out):
    m=mode.model;second=m.backbone.resnet.layer1;third=m.backbone.resnet.layer2
    deblocks=[m.backbone.deblocks[1],m.backbone.deblocks[2]];projection=mode.readout.history
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='training_periodic_deep_refresh_dependency_audit',
        deep_audit='Fixed H baseline; first512 train scans. At periodic deep refresh only, compare current H group16 descriptor to last true deep anchor H descriptor. Own previous confidence aligned from previous frame separately. Candidate M=relativeRMS>.5 or uncovered or confidence>=.1 radius4. Trace both residual stages backwards from projected D output, union decoder2 and stage3 requirements. No mask reuse from stage1. Original full D always feeds baseline. Counterfactual dependencies/error do not alter history. Initial/reset remains full. No regional backend or closed-loop AP/timing claim.')
    write_json(out/'contract.json',contract)
    names=('content_guard','mandatory_only_floor')
    totals={k:dict(macs=0,backbone_macs=0,decode_projection_macs=0,cells=0,error=0.,energy=0.,required_h_cells=0) for k in names}
    state=None;anchor_desc=None;full_cost=None;records=[];forced=periodic=0;exact=0.
    _,stream=loader(train[:a.deep_audit_frames],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));previous=state
            counter=ConvCounter(second,nn.ModuleList([third,*deblocks,projection]))
            try:pred,new,info=mode.step(ego,previous,sample['meta'])
            finally:counter.close()
            if not info['local_partial']:
                current_h=new['spatial']['maps'][0];current_d=new['cache']['value']
                desc=early_descriptor(current_h,1);shape=current_h.shape[-2:]
                if full_cost is None:
                    full=deep_dependencies(second,third,deblocks,projection,torch.ones_like(desc[:,:1],dtype=torch.bool),shape)
                    full_cost={k:full[k] for k in ('macs','backbone_macs','decode_projection_macs')}
                    assert 2*full['backbone_macs']==127166054400
                assert counter.total==full_cost['macs'],(counter.total,full_cost)
                if previous is None or info['reset'] or anchor_desc is None:
                    forced+=1
                    for total in totals.values():
                        for k in full_cost:total[k]+=full_cost[k]
                else:
                    old=previous['cache'];c=current_d.shape[1]
                    aligned,valid=warp_bev(torch.cat((old['value'],anchor_desc),1),old['pose'],sample['meta']['pose'],mode.bounds)
                    confidence,_=warp_bev(previous['spatial']['confidence'],previous['spatial']['pose'],sample['meta']['pose'],mode.bounds)
                    mandatory=(~valid)|(F.max_pool2d((confidence>=.1).float(),9,1,4)>0)
                    candidate=local_change_mask(desc,aligned[:,c:],valid,.5)|mandatory
                    error=(aligned[:,:c]-current_d).square().mean(1,keepdim=True)
                    energy=current_d.square().mean(1,keepdim=True)
                    entry=dict(scan=index+1,anchor_age_s=(sample['meta']['timestamp_us']-old['stamp'])/1e6,modes={})
                    for name,mask in (('content_guard',candidate),('mandatory_only_floor',mandatory)):
                        trace=deep_dependencies(second,third,deblocks,projection,mask,shape)
                        record={k:trace[k] for k in ('macs','backbone_macs','decode_projection_macs','second_output_cells','third_output_cells','second_layers','third_layers')}
                        record.update(cells=int(mask.sum()),required_h_cells=int(trace['required_h'].sum()),
                            error=float((error*~mask).sum()),energy=float((energy*~mask).sum()))
                        for k in totals[name]:totals[name][k]+=record[k]
                        entry['modes'][name]=record
                    records.append(entry);periodic+=1
                anchor_desc=desc
            else:assert counter.total==0
            if index<8:
                other,other_state,_=mode.step(ego,previous,sample['meta'])
                for key in pred:
                    exact=max(exact,float((pred[key]-other[key]).abs().max()));assert torch.equal(pred[key],other[key])
                assert torch.equal(new['cache']['value'],other_state['cache']['value'])
                assert torch.equal(new['spatial']['maps'][0],other_state['spatial']['maps'][0])
            state=new
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='deep_dependency_audit',scans=index+1))
    assert index+1==a.deep_audit_frames and periodic>0
    results={}
    for name,total in totals.items():
        results[name]=dict(mean_deep_refresh_gflops=2*total['macs']/(index+1)/1e9,
            periodic_executed_fraction=(total['macs']-forced*full_cost['macs'])/(periodic*full_cost['macs']),
            periodic_write_fraction=total['cells']/(periodic*100*252),
            periodic_required_h_fraction=total['required_h_cells']/(periodic*100*252),
            retained_relative_rmse=(total['error']/max(total['energy'],1e-12))**.5)
    write_json(out/'deep_audit.json',dict(status='COMPLETED',split='train',scans=index+1,forced_full_frames=forced,
        periodic_refresh_frames=periodic,full_refresh_cost=full_cost,
        baseline_mean_deep_refresh_gflops=2*(forced+periodic)*full_cost['macs']/(index+1)/1e9,
        results=results,exact_replay_max_error=exact,frames=records,
        limitation='Arithmetic potential with complete fresh dependencies, not implemented savings. Mandatory-only is an optimistic cost floor, not a safe deployable selector. Current/full baseline D is used only for discarded error audit; selected D is never fed into history. No closed-loop AP/latency claim.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='deep_dependency_audit',scans=index+1))

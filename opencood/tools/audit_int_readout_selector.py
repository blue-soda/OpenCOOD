"""Train-only selector diagnosis; counterfactual masks NEVER feed recurrence."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import expand_tiles
from opencood.tools.run_int_ego import loader,to_device,write_json


def mask_cost(mask,cache):
    cells=mask.numel();support=F.max_pool2d(mask.float(),3,1,1)>0
    w1=cache.first.weight.numel()+cache.project.weight.numel();w2=cache.second.weight.numel()
    dense=cells*(w1+w2)
    result=dict(write=float(mask.float().mean()),row_fraction=(int(support.sum())*w1+int(mask.sum())*w2)/float(dense))
    for tile in (8,16):
        active=int((F.max_pool2d(mask.float(),tile,tile,ceil_mode=True)>0).sum())
        result['tile{}_fraction'.format(tile)]=active*((tile+2)**2*w1+tile*tile*w2)/float(dense)
    return result


def energy_masks(current,reference,valid,confidence,threshold):
    pool=lambda x:F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
    error=pool((current-reference).square().mean(1,keepdim=True))
    energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
    # Disjoint groups by energy rank, without inventing foreground/background labels.
    order=energy.flatten().argsort();groups=torch.empty_like(order)
    groups[order]=torch.div(torch.arange(order.numel(),device=order.device)*4,order.numel(),rounding_mode='floor')
    guards=F.max_pool2d((confidence>=.1).float(),9,1,4)>0
    values={};quantiles=torch.quantile(energy.flatten(),energy.new_tensor([.25,.5,.75]))
    for name,floor in (('baseline',energy.new_tensor(1e-6)),('floor_q25',quantiles[0].clamp_min(1e-6)),('floor_q50',quantiles[1].clamp_min(1e-6))):
        changed=(error/torch.maximum(energy,floor)).sqrt()>threshold
        request=expand_tiles(changed,4,current.shape[-2:])|~valid
        content=F.max_pool2d(request.float(),5,1,2)>0
        values[name]=(content|guards|~valid,changed)
    return values,groups,energy,error,quantiles,guards


def audit_selector(a,cfg,train,mode,out):
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='training_readout_selector_diagnostic',selector_frames=a.readout_selector_frames,
        selector_audit='First contiguous training scans, fixed period3 baseline recurrence only. Quantile floors q25/q50 of current/retained-reference input energy are ONE-STEP counterfactual masks, not deployed candidates. Both preserve coverage and own-confidence guard. All counterfactual output errors computed from aligned old output vs discarded dense current readout; no candidate state injection, no AP/latency, no claim low energy means background. Rank quartiles describe energy, not object classes. First8 frames require unchanged replay bitwise.')
    write_json(out/'contract.json',contract)
    cache=mode.readout_cache;state=None;frames=[];exact=0.;full=0
    aggregates={name:dict(count=0,write=0.,row_fraction=0.,tile8_fraction=0.,tile16_fraction=0.,
        retained_error=0.,retained_energy=0.,extra_omitted_error=0.,extra_omitted_cells=0,removed_cells=0)
        for name in ('baseline','floor_q25','floor_q50')}
    bins=[dict(tiles=0,changed_tiles=0,energy=0.,input_error=0.) for _ in range(4)]
    _,stream=loader(train[:a.readout_selector_frames],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));detail={};original=cache.step
            def observed(current,prior,meta,reset=False):
                output,new,info=original(current,prior,meta,reset)
                if info['readout_refresh']:return output,new,info
                packed=torch.cat((prior['value'],prior['reference'],prior['confidence']),1)
                aligned,valid=warp_bev(packed,prior['pose'],meta['pose'],cache.bounds)
                co=cache.second.out_channels;ci=current.shape[1]
                old,reference=aligned[:,:co],aligned[:,co:co+ci]
                masks,groups,energy,input_error,quantiles,guards=energy_masks(current,reference,valid,aligned[:,-1:],cache.threshold)
                baseline=masks['baseline'][0]
                assert torch.equal(baseline,cache.last_mask)
                dense=cache.dense(current)
                assert torch.allclose(output.masked_select(baseline),dense.masked_select(baseline),atol=1e-3,rtol=1e-3)
                stale_error=(old-dense).square().mean(1,keepdim=True)
                target_energy=dense.square().mean(1,keepdim=True)
                for group,b in enumerate(bins):
                    selected=groups==group;b['tiles']+=int(selected.sum())
                    b['changed_tiles']+=int(masks['baseline'][1].flatten()[selected].sum())
                    b['energy']+=float(energy.flatten()[selected].sum())
                    b['input_error']+=float(input_error.flatten()[selected].sum())
                detail['energy_quantiles']=quantiles.cpu().tolist();detail['masks']={}
                for name,(mask,_) in masks.items():
                    assert bool((mask|~guards).all()) and bool((mask|valid).all())
                    assert bool((baseline|~mask).all()),'Floor unexpectedly enlarged selector'
                    omitted=baseline&~mask
                    values=mask_cost(mask,cache)
                    values.update(retained_error=float((stale_error*~mask).sum()),
                        retained_energy=float((target_energy*~mask).sum()),
                        extra_omitted_error=float((stale_error*omitted).sum()),
                        extra_omitted_cells=int(omitted.sum()),removed_cells=int(omitted.sum()))
                    detail['masks'][name]=values
                    aggregate=aggregates[name];aggregate['count']+=1
                    for key,value in values.items():aggregate[key]+=value
                return output,new,info
            cache.step=observed;previous=state
            try:pred,new,info=mode.step(ego,previous,sample['meta'])
            finally:cache.step=original
            assert all(torch.isfinite(v).all() for v in pred.values())
            if index<8:
                other,other_state,_=mode.step(ego,previous,sample['meta'])
                for key in pred:
                    exact=max(exact,float((pred[key]-other[key]).abs().max()));assert torch.equal(pred[key],other[key])
                for key in ('value','reference','confidence'):assert torch.equal(new['readout'][key],other_state['readout'][key])
            state=new;full+=int(info['readout_refresh'])
            detail.update(scan=index+1,refresh=info['readout_refresh']);frames.append(detail)
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='readout_selector_audit',scans=index+1))
    assert index+1==a.readout_selector_frames
    results={}
    for name,item in aggregates.items():
        count=item['count'];assert count>0
        results[name]=dict(partial_frames=count,
            **{key:item[key]/count for key in ('write','row_fraction','tile8_fraction','tile16_fraction')},
            overall_row_fraction=(item['row_fraction']+full)/(count+full),
            retained_relative_rmse=(item['retained_error']/max(item['retained_energy'],1e-12))**.5,
            extra_omitted_mse=item['extra_omitted_error']/max(item['extra_omitted_cells'],1),
            removed_fraction=item['removed_cells']/(count*100*252))
    for b in bins:
        b['change_rate']=b['changed_tiles']/max(b['tiles'],1)
        b['mean_energy']=b['energy']/max(b['tiles'],1)
        b['mean_input_error']=b['input_error']/max(b['tiles'],1)
    write_json(out/'selector_audit.json',dict(status='COMPLETED',split='train',scans=index+1,full_frames=full,
        same_input_replay_max_error=exact,energy_rank_quartiles=bins,one_step_counterfactuals=results,frames=frames,
        limitation='Counterfactual masks do not update history. Energy is not a foreground/background label; teacher feature errors do not establish AP. No candidate latency measured, including quantile overhead.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='readout_selector_audit',scans=index+1))

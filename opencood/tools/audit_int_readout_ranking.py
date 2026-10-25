"""Same executed-MAC ceiling, train-only one-step ranking comparison."""
import json
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import expand_tiles
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.tools.run_int_ego import loader,to_device,write_json


from opencood.models.sub_modules.int_budgeted_readout import executed_macs, budgeted_ranking


def audit_ranking(a,cfg,train,mode,out):
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='training_readout_cost_matched_rank_audit',ranking_frames=a.readout_selector_frames,
        ranking_scope='Baseline period3 recurrence only, first training prefix. Relative-input, absolute-input and oracle-output-error tile scores select largest prefix under baseline actual row MAC ceiling, including radius2 content dilation, fixed coverage/own-confidence guards, radius1 intermediate. Binary search is diagnostic, not a latency claim. Oracle is unavailable before deployment compute and is a heuristic reference, not globally optimal selection. Counterfactual masks never change saved history. First8 frames require bitwise uninstrumented replay; no AP or validation tuning.')
    write_json(out/'contract.json',contract)
    cache=mode.readout_cache;state=None;records=[];full=0;exact=0.
    names=('baseline','relative_rank','absolute_rank','oracle_output_rank')
    totals={k:dict(frames=0,macs=0,budget=0,cells=0,xor=0,error=0.,energy=0.,saved_error=0.) for k in names}
    w1=cache.first.weight.numel()+cache.project.weight.numel();w2=cache.second.weight.numel()
    pool=lambda x:F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
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
                mandatory=(F.max_pool2d((~valid).float(),5,1,2)>0)|(F.max_pool2d((aligned[:,-1:]>=.1).float(),9,1,4)>0)
                changed=local_change_mask(current,reference,valid,.5)
                baseline=(F.max_pool2d(changed.float(),5,1,2)>0)|mandatory
                assert torch.equal(baseline,cache.last_mask)
                budget=executed_macs(baseline,w1,w2)
                absolute=pool((current-reference).square().mean(1,keepdim=True))
                energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
                dense=cache.dense(current)
                assert torch.allclose(output.masked_select(baseline),dense.masked_select(baseline),atol=1e-3,rtol=1e-3)
                stale_error=(old-dense).square().mean(1,keepdim=True)
                target_energy=dense.square().mean(1,keepdim=True)
                candidates=dict(baseline=(baseline,dict(macs=budget,budget=budget,budget_fraction=1.,tiles=None)))
                scores=dict(relative_rank=absolute/energy.clamp_min(1e-6),absolute_rank=absolute,oracle_output_rank=pool(stale_error))
                for name,score in scores.items():candidates[name]=budgeted_ranking(score,mandatory,budget,w1,w2)
                detail['candidates']={}
                for name,(mask,stats) in candidates.items():
                    values=dict(stats,cells=int(mask.sum()),xor=int((mask^baseline).sum()),
                        error=float((stale_error*~mask).sum()),energy=float((target_energy*~mask).sum()),
                        saved_error=float((stale_error*mask).sum()))
                    detail['candidates'][name]=values;total=totals[name];total['frames']+=1
                    for key in ('macs','budget','cells','xor','error','energy','saved_error'):total[key]+=values[key]
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
            state=new;full+=int(info['readout_refresh']);detail.update(scan=index+1,refresh=info['readout_refresh']);records.append(detail)
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='readout_ranking_audit',scans=index+1))
    assert index+1==a.readout_selector_frames
    summaries={}
    for name,t in totals.items():
        assert t['frames']>0
        summaries[name]=dict(partial_frames=t['frames'],executed_budget_fraction=t['macs']/max(t['budget'],1),
            mean_partial_write=t['cells']/(t['frames']*100*252),mean_xor_fraction=t['xor']/(t['frames']*100*252),
            retained_relative_rmse=(t['error']/max(t['energy'],1e-12))**.5,
            retained_error_vs_baseline=t['error']/max(totals['baseline']['error'],1e-12),
            captured_stale_error_fraction=t['saved_error']/max(t['saved_error']+t['error'],1e-12))
    write_json(out/'ranking_audit.json',dict(status='COMPLETED',split='train',scans=index+1,full_frames=full,
        exact_replay_max_error=exact,summaries=summaries,frames=records,
        limitation='Counterfactual one-step feature error at no greater MAC budget. Not independently recurrent AP, detector-error importance, deployed speed, or oracle optimality.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='readout_ranking_audit',scans=index+1))

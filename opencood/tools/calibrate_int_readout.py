"""Fit one fixed relative threshold from train-only frozen causal histories."""
import json
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_readout_calibration import fit_threshold,threshold_mask
from opencood.tools.run_int_ego import loader,to_device,write_json


def calibrate_readout(a,cfg,train,mode,out):
    contract=json.loads((out/'contract.json').read_text())
    identity={k:contract[k] for k in ('source','memory','readout','repair','basis','data')}
    contract.update(run_kind='training_fixed_readout_threshold_calibration',
        calibration_scope='First512 training scans, original threshold.5 independent recurrence. Fit ONE scalar to 80% aggregate partial-frame executed row MACs, including halos/guards. No AP, GT, dense teacher or validation data used. Calibrated masks never feed training history. Deployment uses ordinary threshold selection without per-frame sorting/search. Actual closed-loop cost and AP still require full validation.')
    write_json(out/'contract.json',contract)
    cache=mode.readout_cache;state=None;scores=[];mandatory=[];full=0;exact=0.
    pool=lambda x:F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
    _,stream=loader(train[:a.readout_calibrate_frames],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));original=cache.step
            def observed(current,prior,meta,reset=False):
                output,new,info=original(current,prior,meta,reset)
                if not info['readout_refresh']:
                    packed=torch.cat((prior['value'],prior['reference'],prior['confidence']),1)
                    aligned,valid=warp_bev(packed,prior['pose'],meta['pose'],cache.bounds)
                    co=cache.second.out_channels;reference=aligned[:,co:co+current.shape[1]]
                    error=pool((current-reference).square().mean(1,keepdim=True))
                    energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
                    score=(error/energy.clamp_min(1e-6)).sqrt()
                    guard=(F.max_pool2d((~valid).float(),5,1,2)>0)|(F.max_pool2d((aligned[:,-1:]>=.1).float(),9,1,4)>0)
                    assert torch.equal(threshold_mask(score,guard,.5),cache.last_mask)
                    scores.append(score.detach());mandatory.append(guard.detach())
                return output,new,info
            cache.step=observed;previous=state
            try:pred,state,info=mode.step(ego,previous,sample['meta'])
            finally:cache.step=original
            if index<8:
                other,other_state,_=mode.step(ego,previous,sample['meta'])
                for key in pred:
                    exact=max(exact,float((pred[key]-other[key]).abs().max()));assert torch.equal(pred[key],other[key])
                for key in ('value','reference','confidence'):assert torch.equal(state['readout'][key],other_state['readout'][key])
            full+=int(info['readout_refresh'])
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='readout_calibration',scans=index+1))
        assert index+1==a.readout_calibrate_frames and scores
        fit=fit_threshold(torch.cat(scores),torch.cat(mandatory),
            cache.first.weight.numel()+cache.project.weight.numel(),cache.second.weight.numel())
    write_json(out/'calibration.json',dict(status='COMPLETED',split='train',scans=index+1,full_frames=full,
        partial_frames=len(scores),base_threshold=.5,target_ratio=.8,fit=fit,identity=identity,
        exact_replay_max_error=exact,limitation='One-step aggregate calibration only; no per-frame bound or closed-loop AP/latency claim.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='readout_calibration',scans=index+1))

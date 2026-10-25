"""Bounded TRAINING-stream diagnosis; teachers never enter recurrent states."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.tools.run_int_ego import loader,to_device,write_json


def audit_phases(a,cfg,train,modes,out,expected):
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='training_readout_phase_diagnostic',audit_frames=a.readout_audit_frames,
        diagnostic_scope='First contiguous training scans only; no AP, latency or fullval extrapolation. Independently recurrent period3/5, no changes to selection. Extra dense readout is discarded and never injected into history; record content/guard/uncovered masks and retained-output error by deep/readout refresh phase. First8 frames check diagnostic vs unwrapped same-state replay bitwise.')
    write_json(out/'contract.json',contract)
    states={k:None for k in modes};records={k:[] for k in modes};exact=0.
    _,stream=loader(train[:a.readout_audit_frames],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'))
            for name,mode in modes.items():
                cache=mode.readout_cache;original=cache.step;detail={}
                def observed(current,state,meta,reset=False):
                    output,new,info=original(current,state,meta,reset)
                    mask=cache.last_mask
                    dense=cache.dense(current)
                    assert torch.allclose(output.masked_select(mask),dense.masked_select(mask),atol=1e-3,rtol=1e-3)
                    inactive=~mask
                    error=(output-dense).square().mean(1,keepdim=True)
                    energy=dense.square().mean(1,keepdim=True)
                    detail['retained_relative_rmse']=float(((error*inactive).sum()/(energy*inactive).sum().clamp_min(1e-12)).sqrt())
                    detail['retained_mse']=float((error*inactive).sum()/inactive.sum().clamp_min(1))
                    detail['readout_depth']=new['frames']
                    if not info['readout_refresh']:
                        packed=torch.cat((state['value'],state['reference'],state['confidence']),1)
                        aligned,valid=warp_bev(packed,state['pose'],meta['pose'],cache.bounds)
                        co=cache.second.out_channels;ci=current.shape[1]
                        changed=local_change_mask(current,aligned[:,co:co+ci],valid,cache.threshold)
                        content=F.max_pool2d(changed.float(),5,1,2)>0
                        priority=F.max_pool2d((aligned[:,-1:]>=.1).float(),9,1,4)>0
                        assert torch.equal(mask,content|priority|~valid)
                        detail.update(content_fraction=float(changed.float().mean()),
                            expanded_content_fraction=float(content.float().mean()),
                            priority_fraction=float(priority.float().mean()),
                            guard_added_fraction=float((priority&~content).float().mean()),
                            uncovered_fraction=float((~valid).float().mean()))
                    return output,new,info
                cache.step=observed;previous=states[name]
                try:pred,new,info=mode.step(ego,previous,sample['meta'])
                finally:cache.step=original
                assert all(torch.isfinite(x).all() for x in pred.values())
                if index<8:
                    control,other,_=mode.step(ego,previous,sample['meta'])
                    for key in pred:
                        exact=max(exact,float((pred[key]-control[key]).abs().max()))
                        assert torch.equal(pred[key],control[key])
                    for key in ('value','reference','confidence'):
                        assert torch.equal(new['readout'][key],other['readout'][key])
                states[name]=new
                detail.update(scan=index+1,deep_refresh=not info['local_partial'],
                    readout_refresh=info['readout_refresh'],deep_depth=info['anchor_depth'],
                    total_gflops=2*expected(name,info)/1e9,
                    write_fraction=info['readout_write_fraction'],executed_fraction=info['readout_executed_fraction'])
                records[name].append(detail)
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='training_readout_diagnostic',scans=index+1))
    assert index+1==a.readout_audit_frames
    summaries={}
    for name,rows in records.items():
        groups={}
        for deep in (False,True):
            for readout in (False,True):
                values=[r for r in rows if r['deep_refresh']==deep and r['readout_refresh']==readout]
                if not values:continue
                fields=['total_gflops','write_fraction','executed_fraction','retained_relative_rmse','retained_mse']
                if not readout:fields+=['content_fraction','expanded_content_fraction','priority_fraction','guard_added_fraction','uncovered_fraction']
                groups['deep{}_readout{}'.format(int(deep),int(readout))]=dict(count=len(values),
                    **{key:float(np.mean([r[key] for r in values])) for key in fields})
        summaries[name]=dict(frames=len(rows),mean_total_gflops=float(np.mean([r['total_gflops'] for r in rows])),groups=groups)
    write_json(out/'phase_audit.json',dict(status='COMPLETED',split='train',scans=index+1,
        exact_diagnostic_replay_max_error=exact,summaries=summaries,frames=records,
        limitation='Prefix diagnosis, not fullval AP or benchmark. Dense teacher conditioned on each mode own current H/deep path; errors measure readout reuse only. No ground truth or teacher state used in selection.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='training_readout_diagnostic',scans=index+1))

"""Training-only descriptor calibration; dense targets are diagnostic, not inference."""
import json
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_gate import cache_descriptor, descriptor_change
from opencood.tools.run_int_ego import loader, to_device


def replay_gate(records, threshold, max_period=3, max_age=.25):
    """Causal replay: every decision uses the last actual refresh, not last frame."""
    reference=None;skipped=[];errors=[];reasons={k:0 for k in ['initial','period','age','coverage','change','reuse']}
    for index,row in enumerate(records):
        reason='initial' if reference is None or row['reset'] else None
        lag=0 if reference is None else index-reference
        if reason is None:
            if lag>=max_period:reason='period'
            else:
                pair=row['lags'][str(lag)]
                if pair['age']>=max_age:reason='age'
                elif not pair['valid']:reason='coverage'
                elif pair['score']>threshold:reason='change'
        if reason is None:
            skipped.append(index);errors.append(pair['target_relative_error']);reasons['reuse']+=1
        else:
            reference=index;reasons[reason]+=1
    return dict(reuse_fraction=len(skipped)/len(records),counts=reasons,
                reused_target_error_mean=float(np.mean(errors)) if errors else None,
                reused_target_error_p95=float(np.percentile(errors,95)) if errors else None)


def calibrate(a,model,cfg,rows,out,provenance):
    assert a.stage=='calibrate' and provenance['split']=='train'
    chosen=rows[:a.frames];_,stream=loader(chosen,cfg,4,303)
    captured={}
    hooks=[getattr(model.backbone.resnet,'layer%d'%i).register_forward_hook(
        lambda module,inputs,output,index=i:captured.__setitem__(index,output.detach())) for i in [1,2]]
    history=[];state=None;records=[];started=time.monotonic()
    bounds=cfg['model']['args']['lidar_range']
    try:
        with torch.no_grad(),(out/'calibration_frames.jsonl').open('x',buffering=1) as log:
            for index,(sample,row) in enumerate(zip(stream,chosen)):
                meta=sample['meta'];assert meta['frame']==row['frame']
                ego=to_device(sample['ego'],torch.device('cuda'))
                _,state,info=model.step(ego,state,meta)
                if info['reset']:history=[]
                desc=cache_descriptor(captured[1]);target=captured[2]
                assert desc.shape[-2:]==target.shape[-2:]
                record=dict(frame=meta['frame'],segment=meta['segment'],reset=bool(info['reset']),lags={})
                for lag,past in enumerate(reversed(history),1):
                    age=(int(meta['timestamp_us'])-past['time'])/1e6
                    aligned,valid=warp_bev(past['desc'],past['pose'],meta['pose'],bounds)
                    pair=dict(age=age,valid=bool(valid.all()),score=None,target_relative_error=None)
                    if pair['valid']:
                        old,_=warp_bev(past['target'],past['pose'],meta['pose'],bounds)
                        pair.update(score=descriptor_change(desc,aligned),target_relative_error=float(
                            ((target-old).square().mean()/torch.maximum(target.square().mean(),old.square().mean()).clamp_min(1e-6)).sqrt()))
                    record['lags'][str(lag)]=pair
                history.append(dict(desc=desc,target=target,pose=meta['pose'],time=int(meta['timestamp_us'])))
                history=history[-2:]
                records.append(record);log.write(json.dumps(record)+'\n')
                if len(records)%200==0:print('CALIBRATE TRAIN',len(records),'/',len(chosen),flush=True)
    finally:
        for hook in hooks:hook.remove()
    assert len(records)==len(chosen)
    pairs=[p for r in records for p in r['lags'].values() if p['valid'] and p['age']<.25]
    assert pairs,'No eligible training pairs'
    scores=np.array([p['score'] for p in pairs]);errors=np.array([p['target_relative_error'] for p in pairs])
    assert np.isfinite(scores).all() and np.isfinite(errors).all()
    candidates=sorted(set([0.]+np.quantile(scores,np.linspace(0,1,101)).tolist()))
    evaluations=[dict(threshold=t,**replay_gate(records,t)) for t in candidates]
    selected=min(evaluations,key=lambda r:(abs(r['reuse_fraction']-.5),r['threshold']))
    (out/'threshold_candidates.json').write_text(json.dumps(evaluations,indent=2)+'\n')
    correlation=float(np.corrcoef(scores,errors)[0,1]) if scores.std()>0 and errors.std()>0 else None
    return dict(status='COMPLETED',split='train',scans=len(records),eligible_pairs=len(pairs),
                descriptor='groupmean16_avgpool2_block4_relative_rms_p90_v1',max_period=3,max_age=.25,
                target_reuse_fraction=.5,selected=selected,score_target_error_pearson=correlation,
                checkpoint_sha256=provenance['checkpoint']['sha256'],contract=provenance['contract'],
                seconds=time.monotonic()-started,
                caveat='Threshold selected using training descriptor scores and causal reuse rate only; dense feature error is diagnostic, not AP or speed. No validation frames/labels used.')

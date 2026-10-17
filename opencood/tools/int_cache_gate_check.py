"""Check actual no-teacher execution against frozen training replay frame by frame."""
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_cache_repair import PeriodicStageDetector
from opencood.tools.run_int_ego import loader, to_device
from opencood.tools.launch_int_memory import digest


def check_online_gate(a,model,cfg,rows,out):
    policy=json.loads(Path(a.content_calibration).read_text())
    trace_path=Path(a.content_calibration).parent/'calibration_frames.jsonl'
    assert digest(trace_path)==policy['source_trace_sha256']
    trace=[json.loads(s) for s in trace_path.read_text().splitlines()]
    assert a.frames==len(trace)==policy['scans']
    _,stream=loader(rows[:a.frames],cfg,4,303)
    wrapper=PeriodicStageDetector(model,cfg,a.periodic_last,a.max_age,a.content_threshold)
    calls=[0]
    def count_call(module,inputs,output):calls[0]+=1
    hook=model.backbone.resnet.layer2.register_forward_hook(count_call)
    counts=dict(initial=0,period=0,age=0,coverage=0,change=0,reuse=0)
    state=None;reference=None;score_error=0.;seen=0
    try:
        with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
            for index,(sample,row) in enumerate(zip(stream,trace)):
                meta=sample['meta']
                assert meta['frame']==row['frame'] and meta['segment']==row['segment']
                ego=to_device(sample['ego'],torch.device('cuda'))
                before=calls[0];prediction,state,info=wrapper.step(ego,state,meta)
                assert all(torch.isfinite(v).all() for v in prediction.values())
                assert bool(info['reset'])==row['reset']
                reason='initial' if reference is None or row['reset'] else None
                pair=None
                if reason is None:
                    lag=index-reference
                    if lag>=a.periodic_last:reason='period'
                    else:
                        pair=row['lags'][str(lag)]
                        if pair['age']>=a.max_age:reason='age'
                        elif not pair['valid']:reason='coverage'
                        elif pair['score']>a.content_threshold:reason='change'
                assert info['periodic_refresh_reason']==reason,(index,reason,info)
                assert calls[0]-before==int(reason is not None),'Hidden extra stage computation'
                if info['content_score'] is not None:
                    error=abs(info['content_score']-pair['score']);score_error=max(score_error,error)
                    assert error<1e-6,(index,error)
                counts[reason or 'reuse']+=1
                if reason is not None:reference=index
                log.write(json.dumps(dict(frame=meta['frame'],segment=meta['segment'],reason=reason,
                    score=info['content_score'],stage_calls=calls[0]-before,cache_bytes=info['cache_bytes']))+'\n')
                seen+=1
                if seen%200==0:print('CHECK TRAIN',seen,'/',len(trace),flush=True)
    finally:hook.remove()
    assert seen==len(trace) and counts==policy['selected']['counts']
    return dict(status='PASSED',split='train',scans=seen,counts=counts,actual_stage_calls=calls[0],
                max_descriptor_score_error=score_error,cache_bytes=info['cache_bytes'],
                content_calibration_sha256=a.content_calibration_sha256,
                checks=['All frame decisions equal frozen replay','Forward hook proves exactly one stage call on refresh and zero on reuse',
                        'Descriptor score agrees within 1e-6','Finite predictions; training only; no detection labels used'])

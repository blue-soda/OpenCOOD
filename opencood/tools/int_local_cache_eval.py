"""Bounded actual-execution comparison before considering learned repair."""
import json
import subprocess
import sys
import numpy as np
import torch
from opencood.models.sub_modules.int_local_cache import LocalCacheDetector
from opencood.tools.run_int_cache_repair import evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json


def run_local_suite(a,model,cfg,rows,out,policy):
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:raise RuntimeError('INT tests failed')
    threshold=policy['selected']['threshold']
    reference_state=full_state=local_state=None;errors=[];crop_errors=[]
    full=LocalCacheDetector(model,cfg,threshold,period=1)
    local=LocalCacheDetector(model,cfg,threshold,verify=True)
    _,stream=loader(rows[:8],cfg,2,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            expected,reference_state,_=model.step(ego,reference_state,meta)
            pred,full_state,_=full.step(ego,full_state,meta)
            error=max(float((pred[k]-expected[k]).abs().max()) for k in pred)
            assert error<=1e-5;errors.append(error)
            pred,local_state,info=local.step(ego,local_state,meta)
            assert all(torch.isfinite(v).all() for v in pred.values())
            crop_errors.extend(info['local_verification_errors'])
    engineering=dict(status='PASSED',full_refresh_prediction_errors=errors,selected_crop_errors=crop_errors,
                     teacher_scope='Engineering only. Production LocalCacheDetector.verify=False.')
    write_json(out/'engineering.json',engineering)
    reports={}
    for name,period in [('dense_control',1),('local_repair',2)]:
        write_json(out/'status.json',dict(status='RUNNING',phase=name))
        target=out/name;target.mkdir()
        wrapper=LocalCacheDetector(model,cfg,threshold,period=period)
        torch.cuda.reset_peak_memory_stats()
        report=evaluate(a,model,cfg,rows,target,wrapper)
        frames=[json.loads(s) for s in (target/'frames.jsonl').read_text().splitlines()]
        partial=[r for r in frames if r['local_partial']]
        report.update(source_policy=policy,partial_frames=len(partial),
            partial_labeled_frames=sum(r['supervised'] for r in partial),
            actual_reuse_frames=sum(any(s['actual_write_fraction']<1 for s in r['cache_stages'][1:]) for r in frames),
            actual_reuse_labeled_frames=sum(r['supervised'] and any(s['actual_write_fraction']<1 for s in r['cache_stages'][1:]) for r in frames),
            partial_stage_fallbacks=[sum(r['cache_stages'][i]['fallback'] for r in partial) for i in [1,2]],
            partial_mean_write_fraction=[float(np.mean([r['cache_stages'][i]['actual_write_fraction'] for r in partial])) if partial else 1. for i in [1,2]],
            caveats=['Actual cropped execution with tile4 and crop-area>=0.8 full-stage fallback; no inference teacher.',
                     'Stage3 dependencies include actual stage2 writes, including fallback; current stage1/FM/decoder/head stay dense.',
                     'Selected crop outputs are exact relative to possibly approximate upstream inputs, not the original dense teacher.',
                     'State stays anchored at last full refresh; one partial frame maximum, age0.25s.',
                     'Input-area ratios are not MACs or measured latency; timed model includes warp/selection/cache/crops.',
                     'Prefix only, single sequential comparison; no full-validation accuracy conclusion.'])
        write_json(target/'report.json',report);reports[name]=report
    return dict(status='COMPLETED',engineering=engineering,results=reports)

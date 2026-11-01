"""Train-only budget calibration and own-state detector replay, without AP."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_event import EvidenceTraceAnchor,EventAnchoredPyramidDetector
from opencood.models.sub_modules.int_event_refresh import replay_refresh_trace
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def capture(base,ego,state,meta):
    inputs=[];hook=base.core.observation.register_forward_pre_hook(lambda m,x:inputs.append(x[0]))
    try:_,state,_=base.step(ego,state,meta)
    finally:hook.remove()
    return inputs[0],state


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    source=Path(a.strict_source);source_report=json.loads(source.read_text())
    assert source_report['status']=='COMPLETED' and source_report['split']=='train'
    assert source_report['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    assert len(gold)==256
    trace_model=EvidenceTraceAnchor(base);trace=[];base_state=state=None
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),source_sha256=digest(source),
        split='train',frames=256,ap_evaluated=False,trained=False,tau=.2,reference_dt=.1,max_age=.25,
        evidence='L1(current recurrent coarse - aligned previous coarse) on valid cells / L1(current coarse) on valid cells, epsilon1e-6.',
        calibration='129 log2 thresholds from max_evidence*2^-12 to max_evidence*2^4; minimize absolute total-refresh difference; ties prefer under-budget then highest threshold.',
        coverage='Each candidate looks up coverage from its actual anchor pose; forced resets/age/coverage included in total budget.',
        storage='22,579,200 tensor bytes plus one Python scalar membrane; metadata excluded consistently with anchor baseline.',
        boundary='Hybrid ANN scalar event controller, not spiking feature network. No AP or latency claims.'))
    _,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'trace.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
            x,base_state=capture(base,ego,base_state,meta)
            (_,state,info),cost=measured_step(trace_model,x,state,meta)
            assert meta['frame']==gold[index]['frame']
            expected=gold[index]['arms']['period2_strict']
            assert info['late_refresh']==expected['late_refresh'] and cost==expected['core_conv_flops']
            assert all(torch.equal(state[k],base_state[k]) for k in ('value','coarse'))
            stamp=int(meta['timestamp_us']);coverage={}
            bounds=block_lidar_range(base.bounds,x.shape[-2:],state['anchor']['value'].shape[-2:],8)
            ones=torch.ones_like(state['anchor']['value'][:,:1])
            for past in range(index-1,-1,-1):
                row=trace[past]
                if row['segment']!=meta['segment'] or (stamp-row['timestamp_us'])/1e6>=.25-1e-6:break
                _,valid=warp_bev(ones,row['pose'],meta['pose'],bounds)
                coverage[str(past)]=float((~valid).float().mean())
            row=dict(frame=meta['frame'],timestamp_us=stamp,segment=meta['segment'],reset=info['reset'],
                pose=np.asarray(meta['pose']).tolist(),evidence=info['refresh_evidence'],coverage_by_anchor=coverage,
                strict_refresh=info['late_refresh'],strict_core_flops=cost)
            trace.append(row);log.write(json.dumps(row)+'\n')
            if len(trace)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='trace',scans=len(trace)))
    assert len(trace)==256
    target=sum(r['strict_refresh'] is not None for r in trace);scale=max(max(r['evidence'] for r in trace),1e-6)
    thresholds=[scale*2**(-12+i/8.) for i in range(129)];chosen={};plans={};search={}
    for mode in ('threshold','continuous','event'):
        candidates=[]
        for threshold in thresholds:
            plan=replay_refresh_trace(trace,mode,threshold);count=sum(r['refresh'] is not None for r in plan)
            candidates.append(dict(threshold=threshold,total_refresh=count,delta=count-target))
        best=min(candidates,key=lambda r:(abs(r['delta']),r['delta']>0,-r['threshold']))
        chosen[mode]=best;search[mode]=candidates;plans[mode]=replay_refresh_trace(trace,mode,best['threshold'])
    write_json(out/'calibration.json',dict(target_total_refresh=target,chosen=chosen,search=search,plans=plans))
    models={mode:EventAnchoredPyramidDetector(base,mode,v['threshold']) for mode,v in chosen.items()}
    states={mode:None for mode in models};base_state=None;records=[]
    _,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'replay.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            assert meta['frame']==trace[index]['frame'];record=dict(frame=meta['frame'],arms={})
            for mode,model in models.items():
                (pred,states[mode],info),cost=measured_step(model,x,states[mode],meta);expected=plans[mode][index]
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert all(torch.equal(states[mode][k],base_state[k]) for k in ('value','coarse'))
                assert math.isclose(info['refresh_evidence'],trace[index]['evidence'],rel_tol=1e-6,abs_tol=1e-7)
                assert info['late_refresh']==expected['refresh']
                assert info['anchor_timestamp_us']==trace[expected['anchor_index']]['timestamp_us']
                assert math.isclose(states[mode]['controller']['membrane'],expected['membrane'],rel_tol=1e-6,abs_tol=1e-7)
                if info['late_refresh'] is None:assert info['uncovered_fraction']==0 and info['anchor_age_before_refresh']<.25
                record['arms'][mode]=dict(core_flops=cost,membrane=states[mode]['controller']['membrane'],**info)
            records.append(record);log.write(json.dumps(record)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='own_state_replay',scans=len(records)))
    assert len(records)==256
    dense=max(r['strict_core_flops'] for r in trace)*256;strict_work=sum(r['strict_core_flops'] for r in trace)
    metrics={}
    for mode in models:
        arms=[r['arms'][mode] for r in records];count=sum(r['late_refresh'] is not None for r in arms)
        work=sum(r['core_flops'] for r in arms)
        assert count==chosen[mode]['total_refresh']
        if count==target:assert work==strict_work
        metrics[mode]=dict(total_refresh=count,budget_delta=count-target,core_ratio=work/dense,
            forced=sum(r['controller']['forced'] is not None for r in arms),spikes=sum(r['controller']['spike'] for r in arms),
            evidence_refresh=sum(r['late_refresh']=='evidence' for r in arms),tensor_cache_bytes=arms[-1]['cache_bytes'],scalar_membrane_count=1)
    write_json(out/'report.json',dict(status='COMPLETED',split='train',scans=256,checkpoint_sha256=digest(Path(a.checkpoint)),
        target_total_refresh=target,strict_core_ratio=strict_work/dense,chosen=chosen,metrics=metrics,
        offline_online_schedule_and_membrane_match=True,coarse_states_match_dense_all=True,ap_evaluated=False,timing_valid=False,
        caveat='Matched training refresh counts are calibration, not generalization or matched validation cost. No SNN feature or AP claim.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','strict-source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h calibration budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.calibrate_int_event_refresh']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

"""Exclusive rotated-order timing of dense, anchor-period1 and anchor-period2."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.benchmark_int_late_cache import check_exclusive


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    occupancy=[check_exclusive(a.gpu)]
    source=Path(a.source_report);report=json.loads(source.read_text())
    assert report['status']=='COMPLETED' and report['period']==2 and report['coverage']=='strict'
    assert report['scans']==5794 and report['labels']==1738 and report['max_age']==.25
    assert report['checkpoint_sha256']==digest(Path(a.checkpoint))
    with (source.parent/'frames.jsonl').open() as stream:reference=[json.loads(next(stream)) for _ in range(a.frames)]
    a.mode='gru';a.stage='pilot';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    models=dict(dense=base,period1=AnchoredLatePyramidDetector(base,period=1),
                period2=AnchoredLatePyramidDetector(base,period=2,coverage='strict',max_age=.25))
    for module in base.modules():assert not module._forward_hooks and not module._forward_pre_hooks
    names=list(models);orders=[names[i:]+names[:i] for i in range(3)];timings={name:[] for name in names};rounds=[]
    grouped={};all_timings={name:[] for name in names}
    write_json(out/'contract.json',dict(checkpoint_sha256=report['checkpoint_sha256'],source_report_sha256=digest(source),
        frames=a.frames,full_stream=a.frames==5794,rounds=3,orders=orders,warmup=16,period2_coverage='strict',max_age=.25,
        source_ap70=report['metrics']['bev_ap70'],source_ap_scope='R072 frozen full validation, no AP recomputation here',
        timing_scope='Synchronized model.step wall time includes Pillar, warp, refresh decisions, dense/skip deep, decoder and heads; excludes IO/H2D/postprocess',
        instrumentation='No forward/count hooks; comparisons and occupancy sampling outside timed interval'))
    with torch.no_grad(),(out/'timings.jsonl').open('x',buffering=1) as log:
        for repeat,order in enumerate(orders):
            states={name:None for name in names};local={name:[] for name in names}
            _,stream=loader(rows[:a.frames],cfg,2,303)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];preds={};infos={};elapsed={}
                assert meta['frame']==reference[index]['frame']
                assert bool(meta['supervised'])==bool(reference[index]['supervised'])
                for name in order:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    preds[name],states[name],infos[name]=models[name].step(ego,states[name],meta)
                    torch.cuda.synchronize();elapsed[name]=1000*(time.perf_counter()-tick)
                    all_timings[name].append(elapsed[name])
                    if index>=16:local[name].append(elapsed[name]);timings[name].append(elapsed[name])
                assert all(torch.equal(preds['period1'][k],preds['dense'][k]) for k in preds['dense'])
                actual=infos['period2'];expected=reference[index]
                for key in ('reset','late_refresh','anchor_timestamp_us','cache_bytes'):
                    assert actual[key]==expected[key],(key,meta['frame'])
                assert actual['late_refresh'] is not None or actual['uncovered_fraction']==0
                assert all(torch.isfinite(v).all() for v in preds['period2'].values())
                group=actual['late_refresh'] or 'reuse'
                if index>=16:
                    bucket=grouped.setdefault(group,{name:[] for name in names})
                    for name in names:bucket[name].append(elapsed[name])
                log.write(json.dumps(dict(round=repeat,frame=meta['frame'],order=order,warmup=index<16,ms=elapsed,
                    period1_dense_exact=True,period2_schedule_matches=True,late_refresh=actual['late_refresh'],
                    reset=actual['reset'],anchor_timestamp_us=actual['anchor_timestamp_us'],
                    uncovered_fraction=actual['uncovered_fraction'],supervised=bool(meta['supervised']),
                    cache_bytes=actual['cache_bytes']))+'\n')
                if (index+1)%64==0:
                    occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
                    write_json(out/'status.json',dict(status='RUNNING',phase='timing',round=repeat,scans=index+1,total=a.frames))
            assert index+1==a.frames
            rounds.append({name:float(np.mean(local[name])) for name in names})
            occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
    summary={name:dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values,95)),samples=len(values)) for name,values in timings.items()}
    stratified={group:{name:dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values,95)),samples=len(values)) for name,values in arms.items()}
        for group,arms in grouped.items()}
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=True,frames=a.frames,full_stream=a.frames==5794,
        rounds=rounds,timings=summary,refresh_groups=stratified,
        including_warmup={name:dict(mean_ms=float(np.mean(values)),samples=len(values)) for name,values in all_timings.items()},
        checkpoint_sha256=report['checkpoint_sha256'],period1_dense_exact_all=True,period2_schedule_matches_full_validation=True,
        period2_to_dense_time=summary['period2']['mean_ms']/summary['dense']['mean_ms'],
        period2_to_period1_time=summary['period2']['mean_ms']/summary['period1']['mean_ms'],cache_bytes=report['cache_bytes'],
        caveats=[('Full-stream model timing, not end-to-end latency.' if a.frames==5794 else 'Prefix model timing, not full-stream/end-to-end latency.')+
                 ' Same GPU, shared frozen weights, independent state per arm.',
                 'Exclusivity sampled every 64 frames and round boundaries; not continuous monitoring.',
                 'Period1 outputs match dense; period2 follows full-validation schedule, its AP comes from R072 not this benchmark.',
                 'Warmup first16 frames each round excluded; IO/H2D/postprocess and comparisons excluded from timings.']))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source-report'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--frames',type=int,default=512);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not (64<=a.frames<=512 or a.frames==5794):p.error('64 to 512 prefix frames or exactly5794 full-stream frames allowed')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h anchor timing budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader']).decode().strip():
        raise RuntimeError('Benchmark requires an idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.benchmark_int_late_anchor']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

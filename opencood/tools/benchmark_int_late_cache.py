"""Same-GPU no-hook model timing, after exact compact-cache engineering."""
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
from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.check_int_pyramid_cache import measured_step


def check_exclusive(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader']).decode().strip()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).decode().splitlines()
    pids=[int(line.split(',')[0]) for line in lines if line.split(',')[-1].strip()==uuid]
    if any(pid!=os.getpid() for pid in pids):raise RuntimeError('GPU contention: '+str(pids))
    return dict(unix=time.time(),uuid=uuid,pids=pids)


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    occupancy=[check_exclusive(a.gpu)]
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('Tests failed')
    completed=json.loads(Path(a.source_report).read_text())
    assert completed['status']=='COMPLETED' and completed['policy']=='late_only'
    assert completed['scans']==5794 and completed['labels']==1738
    assert completed['checkpoint_sha256']==digest(Path(a.checkpoint))
    a.mode='gru';a.stage='pilot';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    options=dict(force_full_blocks=list(range(9)),fraction=.1,tile=4,max_cache_age=.5,execution='coalesced')
    original=BlockCachedPyramidDetector(base,**options)
    compact=BlockCachedPyramidDetector(base,compact_dense=True,**options)
    models=dict(dense=base,original=original,compact=compact)
    states={name:None for name in models};engineering=[]
    _,stream=loader(rows[:32],cfg,2,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured=[]
            hook=base.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:_,states['dense'],_=base.step(ego,states['dense'],meta)
            finally:hook.remove()
            x=captured[0]
            (expected,states['original'],oi),oc=measured_step(original,x,states['original'],meta)
            (actual,states['compact'],ci),cc=measured_step(compact,x,states['compact'],meta)
            assert all(torch.equal(expected[k],actual[k]) for k in expected)
            assert all(torch.equal(states['original'][k],states['compact'][k]) for k in base.state_keys)
            for old,new in zip(states['original']['blocks'][9:],states['compact']['blocks'][9:]):
                assert all(torch.equal(old[k],new[k]) for k in old)
            assert oc==cc and ci['cache_bytes']<oi['cache_bytes']
            engineering.append(dict(frame=meta['frame'],predictions_exact=True,deep_cache_exact=True,
                original_core_flops=oc,compact_core_flops=cc,original_cache_bytes=oi['cache_bytes'],
                compact_cache_bytes=ci['cache_bytes'],compact_state_bytes=ci['state_bytes']))
    write_json(out/'engineering.json',dict(status='PASSED',records=engineering,scope='32 real-frame own-state parity; no full compact AP recomputation'))
    for m in base.modules():
        assert not m._forward_hooks and not m._forward_pre_hooks
    occupancy.append(check_exclusive(a.gpu))
    names=list(models);timings={name:[] for name in names};rounds=[]
    write_json(out/'contract.json',dict(checkpoint_sha256=completed['checkpoint_sha256'],frames=a.frames,rounds=3,
        excluded_warmup_frames=16,orders=[names[i:]+names[:i] for i in range(3)],
        scope='Synchronized wall time of model.step including Pillar/warp/selectors/cache/crops/decoder/heads; excludes input preparation/H2D/postprocess',
        instrumentation='No forward/counting hooks; correctness comparisons and nvidia-smi outside timed interval',
        caveats=['Prefix benchmark, not full-stream latency.','Exclusivity sampled every 64 frames and round boundaries, not continuously monitored.']))
    with torch.no_grad(),(out/'timings.jsonl').open('x',buffering=1) as log:
        for repeat in range(3):
            states={name:None for name in names};order=names[repeat:]+names[:repeat]
            local={name:[] for name in names};_,stream=loader(rows[:a.frames],cfg,2,303)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];outputs={};elapsed={}
                for name in order:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    outputs[name],states[name],_=models[name].step(ego,states[name],meta)
                    torch.cuda.synchronize();elapsed[name]=1000*(time.perf_counter()-tick)
                    if index>=16:timings[name].append(elapsed[name]);local[name].append(elapsed[name])
                assert all(torch.equal(outputs['original'][k],outputs['compact'][k]) for k in outputs['original'])
                log.write(json.dumps(dict(round=repeat,frame=meta['frame'],order=order,warmup=index<16,ms=elapsed,compact_exact=True))+'\n')
                if (index+1)%64==0:
                    occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
                    write_json(out/'status.json',dict(status='RUNNING',phase='timing',round=repeat,scans=index+1,total=a.frames))
            rounds.append({name:float(np.mean(local[name])) for name in names})
            occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
    summary={name:dict(mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples=len(values)) for name,values in timings.items()}
    report=dict(status='COMPLETED',timing_valid=True,frames=a.frames,rounds=rounds,timings=summary,
        compact_predictions_exact_all=True,checkpoint_sha256=completed['checkpoint_sha256'],
        original_cache_bytes=engineering[-1]['original_cache_bytes'],compact_cache_bytes=engineering[-1]['compact_cache_bytes'],
        compact_state_bytes=engineering[-1]['compact_state_bytes'],compact_to_dense_time=summary['compact']['mean_ms']/summary['dense']['mean_ms'],
        compact_to_original_time=summary['compact']['mean_ms']/summary['original']['mean_ms'],
        caveats=['Same-card rotated-order prefix model timing; not end-to-end or full-stream latency.',
                 'AP belongs to original R068 full validation; compact parity checked on the engineering/benchmark prefix only.',
                 'Shared frozen weights, independent states; comparisons, IO, H2D and postprocess excluded from timed intervals.'])
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source-report'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--frames',type=int,default=256);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 64<=a.frames<=512:p.error('Benchmark budget: 64 to 512 frames')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h benchmark budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader']).decode().strip():
        raise RuntimeError('Benchmark requires an idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.benchmark_int_late_cache']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

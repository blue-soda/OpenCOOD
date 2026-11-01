"""Exclusive full-stream timing: dense, strict cache, statistics-free count.

Reference-coded full-stream prediction parity is checked outside timed windows
in round zero. All arms have independent state and rotate first/second/third.
"""
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
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor
from opencood.models.sub_modules.int_count_runtime import runtime_count_anchor
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.benchmark_int_late_cache import check_exclusive


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    occupancy=[check_exclusive(a.gpu)]
    src=Path(a.readout_report);prior=json.loads(src.read_text());policy=json.loads((src.parent/'policy.json').read_text())
    strict=json.loads(Path(a.strict_report).read_text())
    assert prior['status']==strict['status']=='COMPLETED' and prior['scans']==strict['scans']==5794
    assert prior['labels']==strict['labels']==1738
    assert prior['checkpoint_sha256']==strict['checkpoint_sha256']==digest(Path(a.checkpoint))
    assert prior['scale_sha256']==policy['scale_sha256'] and policy['steps']==8 and policy['initial_membrane_units']==32768
    assert strict['period']==2 and strict['coverage']=='strict' and strict['max_age']==.25
    gold=[json.loads(x) for x in (src.parent/'frames.jsonl').read_text().splitlines()]
    a.mode='gru';a.stage='pilot';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    reference=readout_coded_anchor(base,policy['scales'],'count',8,True,32768)
    models=dict(dense=base,strict=AnchoredLatePyramidDetector(base),count=runtime_count_anchor(base,policy['scales']))
    for model in models.values():
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    names=list(models);orders=[names[i:]+names[:i] for i in range(3)]
    timings={name:[] for name in names};rounds=[]
    write_json(out/'contract.json',dict(readout_report_sha256=digest(src),strict_report_sha256=digest(Path(a.strict_report)),
        checkpoint_sha256=prior['checkpoint_sha256'],scale_sha256=prior['scale_sha256'],frames=5794,rounds=3,orders=orders,warmup=16,
        parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        gpu=torch.cuda.get_device_name(0),precision='FP32, TF32 disabled',
        scope='Synchronized model.step; excludes IO/H2D/postprocess; no hooks or codec diagnostic reductions in timed arms.',
        parity='All5794 frames in first round compare original count predictions/state outside timing; subsequent rounds match strict refresh/state.',
        ap_sources_only=dict(strict=strict['metrics'],count=prior['metrics']),
        note='Dense Conv execution remains unchanged. Not sparse event timing or neuromorphic hardware.'))
    with torch.no_grad(),(out/'timings.jsonl').open('x',buffering=1) as log:
        for repeat,order in enumerate(orders):
            states={name:None for name in names};reference_state=None;local={name:[] for name in names}
            _,stream=loader(rows,cfg,2,303)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];preds={};infos={};elapsed={}
                assert meta['frame']==gold[index]['frame'] and meta['segment']==gold[index]['segment']
                for name in order:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    preds[name],states[name],infos[name]=models[name].step(ego,states[name],meta)
                    torch.cuda.synchronize();elapsed[name]=1000*(time.perf_counter()-tick)
                    if index>=16:local[name].append(elapsed[name]);timings[name].append(elapsed[name])
                for name in ('strict','count'):
                    for key in ('reset','late_refresh','anchor_timestamp_us','cache_bytes'):
                        assert infos[name][key]==gold[index][key]
                    assert all(torch.equal(states[name][k],states['strict'][k]) for k in ('value','coarse'))
                    assert torch.equal(states[name]['anchor']['value'],states['strict']['anchor']['value'])
                    assert all(torch.isfinite(v).all() for v in preds[name].values())
                if repeat==0:
                    ref,reference_state,_=reference.step(ego,reference_state,meta)
                    assert all(torch.equal(ref[k],preds['count'][k]) for k in ref)
                    assert all(torch.equal(reference_state[k],states['count'][k]) for k in ('value','coarse'))
                    assert torch.equal(reference_state['anchor']['value'],states['count']['anchor']['value'])
                log.write(json.dumps(dict(round=repeat,frame=meta['frame'],order=order,warmup=index<16,ms=elapsed,
                    late_refresh=infos['count']['late_refresh'],cache_equal=True,reference_prediction_exact=True if repeat==0 else None))+'\n')
                if repeat==0 and index==31:write_json(out/'engineering.json',dict(status='PASSED',frames=32,predictions_states_exact=True,no_timed_hooks=True))
                if (index+1)%64==0:
                    occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
                    write_json(out/'status.json',dict(status='RUNNING',phase='timing',round=repeat,scans=index+1,total=5794))
            assert index+1==5794
            rounds.append({name:float(np.mean(local[name])) for name in names})
            occupancy.append(check_exclusive(a.gpu));write_json(out/'occupancy.json',occupancy)
    summary={name:dict(mean_ms=float(np.mean(x)),p50_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),samples=len(x)) for name,x in timings.items()}
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=True,full_stream=True,frames=5794,rounds=rounds,timings=summary,
        checkpoint_sha256=prior['checkpoint_sha256'],scale_sha256=prior['scale_sha256'],reference_exact_frames=5794,
        count_to_strict_time=summary['count']['mean_ms']/summary['strict']['mean_ms'],
        count_to_dense_time=summary['count']['mean_ms']/summary['dense']['mean_ms'],
        parameters=sum(p.numel() for p in models['count'].parameters()),
        core_conv_flops_source=strict['core_conv_flops_mean'],ap_source=prior['metrics'],
        caveats=['AP inherited only after full-stream prediction/state parity; not re-evaluated in timing.',
            'Three rotated rounds; first16 warmup excluded each round; model-only, not end-to-end.',
            'Original count reference comparison only in round0 and outside each timed interval.',
            'GPU exclusivity sampled every64 frames/boundaries; no continuous monitoring or hardware energy claim.']))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','checkpoint','readout-report','strict-report','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h count timing budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader']).decode().strip():
        raise RuntimeError('Timing requires an idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.benchmark_int_count_readout']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

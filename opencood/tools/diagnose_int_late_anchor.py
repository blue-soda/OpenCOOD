"""Training-prefix whole-context reuse diagnostic; no latency or tuning claims."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_pyramid_anchor_patch import PatchedLatePyramidDetector
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    if a.suite=='patch':
        gate=Path(a.source_gate);passed=json.loads(gate.read_text())
        assert passed['status']=='COMPLETED' and passed['scans']==256 and passed['split']=='train'
        assert passed['same_mask_parity_all'] and passed['period1_dense_exact_all']
        assert passed['checkpoint_sha256']==digest(Path(a.checkpoint))
        tests=(gate.parent/'unit_tests.log').read_text()
        assert 'Ran 296 tests' in tests and tests.rstrip().endswith('OK')
        write_json(out/'engineering.json',dict(status='PASSED',source_gate=str(gate),source_sha256=digest(gate),
            scope='Unchanged patch model; reuse R075 296 tests and256 real-frame parity/cost gate'))
    else:
        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/'unit_tests.log').write_bytes(tests.stdout)
        if tests.returncode:raise RuntimeError('Tests failed')
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    configs=dict(dense=dict(period=1,coverage='strict'),period2_strict=dict(period=2,coverage='strict'),
                 period2_zero=dict(period=2,coverage='zero'),period3_zero=dict(period=3,coverage='zero'))
    if a.suite=='patch':del configs['period3_zero']
    models={name:AnchoredLatePyramidDetector(base,**cfg_) for name,cfg_ in configs.items()}
    if a.suite=='patch':
        models['period2_patch']=PatchedLatePyramidDetector(base)
        configs['period2_patch']=dict(period=2,coverage='transient_patch',tile=1)
    states={name:None for name in models};base_state=None;records=[]
    ap={name:{.7:dict(tp=[],fp=[],gt=0,score=[])} for name in models}
    write_json(out/'contract.json',dict(split='train',frames=a.frames,checkpoint_sha256=digest(Path(a.checkpoint)),
        candidates=configs,max_age=.25,trained=False,suite=a.suite,
        policy='Fine/coarse dense every frame. Entire deep encoder/update refresh on period/age/reset; persistent deep anchor never rewarped in storage.',
        coverage='strict fully refreshes on new-view cells; zero retains zero fill; patch repairs only missing cells for current readout without overwriting anchor.',
        scope='Fixed training-prefix diagnostic; no independent AP, GPU speedup, SNN or energy claim'))
    dataset,stream=loader(rows[:a.frames],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured=[]
            hook=base.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:expected,base_state,_=base.step(ego,base_state,meta)
            finally:hook.remove()
            x=captured[0];entry=dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],arms={})
            predictions={}
            for name,model in models.items():
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta)
                predictions[name]=pred
                if name=='dense':
                    assert all(torch.equal(pred[k],expected[k]) for k in pred)
                    assert torch.equal(states[name]['anchor']['value'],base_state['deep'])
                if info['late_refresh'] is None:
                    assert states[name]['anchor']['timestamp_us']<int(meta['timestamp_us'])
                entry['arms'][name]=dict(core_conv_flops=cost,**info)
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],.7)
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if len(records)%32==0:
                write_json(out/'status.json',dict(status='RUNNING',phase='train_prefix',scans=len(records),total=a.frames))
                print('ANCHOR',len(records),'/',a.frames,flush=True)
    assert len(records)==a.frames
    write_json(out/'ap_records.json',ap)
    dense_work=sum(r['arms']['dense']['core_conv_flops'] for r in records)
    report=dict(status='COMPLETED',split='train',scans=len(records),labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),dense_period1_exact_all=True,suite=a.suite,
        metrics={name:dict(ap70=eval_utils.calculate_ap(ap[name],.7)[0],
            core_conv_ratio=sum(r['arms'][name]['core_conv_flops'] for r in records)/dense_work,
            reuse_frames=sum(r['arms'][name]['late_refresh'] is None for r in records),
            whole_reuse_frames=sum(r['arms'][name]['late_refresh'] is None and not (r['arms'][name].get('repair') or {}).get('active',False) for r in records),
            patch_frames=sum((r['arms'][name].get('repair') or {}).get('active',False) for r in records),
            refresh_counts={str(reason):sum(r['arms'][name]['late_refresh']==reason for r in records)
                for reason in sorted({r['arms'][name]['late_refresh'] for r in records},key=str)},
            cache_bytes=records[-1]['arms'][name]['cache_bytes'],
            uncovered_fraction_mean=sum(r['arms'][name]['uncovered_fraction'] for r in records)/len(records)) for name in models},
        caveats=['Train prefix only, not full validation; candidate boundaries predeclared.',
                 'Zero coverage policy is explicit approximation, not boundary repair; deep stale context can miss dynamic/new-view evidence.',
                 'Conv hooks include actual refreshed deep work and full decoder, exclude Pillar and nonconv overhead; no timing claim.',
                 'Fixed-anchor refresh also changes recurrent history alignment semantics; not a pure execution optimization.'])
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--frames',type=int,default=256);p.add_argument('--worker',action='store_true')
    p.add_argument('--suite',choices=['anchor','patch'],default='anchor');p.add_argument('--source-gate')
    a=p.parse_args();out=Path(a.output)
    if not 32<=a.frames<=256:p.error('Training diagnostic budget: 32 to 256 frames')
    if a.suite=='patch' and not a.source_gate:p.error('Patch diagnostic requires source engineering gate')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h anchor diagnostic budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('Insufficient free GPU memory')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.diagnose_int_late_anchor']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

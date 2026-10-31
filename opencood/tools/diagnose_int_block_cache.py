"""Training-prefix causal-chain interventions; oracle history is diagnostic only."""
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
from opencood.models.sub_modules.int_pyramid_block_cache import BlockCachedPyramidDetector
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def relative_mse(actual,reference):
    return float((actual-reference).square().mean()/reference.square().mean().clamp_min(1e-8))


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('Tests failed')
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    full=BlockCachedPyramidDetector(base,fraction=1)
    forced=dict(independent=[],updates_only=[i for i in range(15) if i not in (3,8,14)],
                encoders_only=[3,8,14],late_only=list(range(9)),oracle_previous_dense=[])
    models={name:BlockCachedPyramidDetector(base,fraction=.1,tile=4,max_cache_age=.5,
            execution='coalesced',force_full_blocks=indices) for name,indices in forced.items()}
    states={name:None for name in models};dense_state=full_state=None
    names=['dense']+list(models)
    ap={name:{.7:dict(tp=[],fp=[],gt=0,score=[])} for name in names};records=[]
    write_json(out/'contract.json',dict(split='train',prefix=a.frames,checkpoint_sha256=digest(Path(a.checkpoint)),
        fraction=.1,tile=4,max_cache_age=.5,forced_full_blocks=forced,
        oracle='oracle_previous_dense receives the exact previous full-cache state each frame; offline, nondeployable, not efficiency evidence',
        purpose='Separate within-frame cache boundaries from accumulated mixed-state error; no validation tuning, no training'))
    dataset,stream=loader(rows[:a.frames],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured=[]
            hook=base.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:dense,dense_state,_=base.step(ego,dense_state,meta)
            finally:hook.remove()
            x=captured[0];previous_full=full_state
            exact,full_state,full_info=full.step_spatial(x,full_state,meta)
            assert all(torch.equal(exact[k],dense[k]) for k in dense)
            predictions=dict(dense=dense);entry=dict(frame=meta['frame'],segment=meta['segment'],
                supervised=meta['supervised'],reset=full_info['reset'],full_refresh_exact=True,arms={})
            for name,model in models.items():
                previous=previous_full if name=='oracle_previous_dense' else states[name]
                pred,new,info=model.step_spatial(x,previous,meta)
                if name!='oracle_previous_dense':states[name]=new
                predictions[name]=pred
                entry['arms'][name]=dict(psm_relative_mse=relative_mse(pred['psm'],dense['psm']),
                    rm_relative_mse=relative_mse(pred['rm'],dense['rm']),
                    block_relative_mse=[relative_mse(c['value'],d['value']) for c,d in zip(new['blocks'],full_state['blocks'])],
                    write_fractions=[s['actual_write_fraction'] for s in info['cache_stages']])
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data);entry['detections']={}
                for name,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    entry['detections'][name]=0 if boxes is None else len(boxes)
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],.7)
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if len(records)%32==0:
                write_json(out/'status.json',dict(status='RUNNING',phase='train_prefix',scans=len(records),total=a.frames))
                print('DIAGNOSTIC',len(records),'/',a.frames,flush=True)
    partial=[r for r in records if not r['reset']]
    report=dict(status='COMPLETED',split='train',scans=len(records),labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),full_refresh_all_exact=True,
        ap70={name:eval_utils.calculate_ap(ap[name],.7)[0] for name in names},
        partial_means={name:dict(psm_relative_mse=sum(r['arms'][name]['psm_relative_mse'] for r in partial)/len(partial),
            block_relative_mse=[sum(r['arms'][name]['block_relative_mse'][i] for r in partial)/len(partial) for i in range(15)]) for name in models},
        caveats=['Training prefix diagnostic only; no full-stream AP, generalization or latency claim.',
                 'Force-full interventions change compute and information together, not matched-budget comparisons.',
                 'Oracle previous dense state is nondeployable; improvements cannot isolate warp drift from all mixed-state feedback errors.',
                 'No weights or selector thresholds tuned; dense and all own-state arms have separate histories.'])
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--frames',type=int,default=256);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 32<=a.frames<=256:p.error('Bounded train diagnostic: 32 to 256 scans')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h diagnostic budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('Insufficient free GPU memory')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.diagnose_int_block_cache']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

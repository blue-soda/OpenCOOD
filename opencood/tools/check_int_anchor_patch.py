"""Training-stream geometry/cost gate for transient anchor boundary repair."""
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


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    torch.cuda.set_per_process_memory_fraction(.3)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    models=dict(period1=AnchoredLatePyramidDetector(base,period=1),strict=AnchoredLatePyramidDetector(base),
                patch=PatchedLatePyramidDetector(base),reference=PatchedLatePyramidDetector(base,dense_reference=True))
    states={name:None for name in models};base_state=None;records=[]
    write_json(out/'contract.json',dict(split='train',frames=a.frames,checkpoint_sha256=digest(Path(a.checkpoint)),
        period=2,max_age=.25,tile=1,trained=False,ap_evaluated=False,timing_valid=False,
        policy='Only missing coverage cells repaired for current readout, persistent anchor untouched until scheduled full refresh.',
        engineering='Full period1 parity; own-state crop versus same-mask dense parity; actual Conv hooks include full decoder and repeated halos.',
        boundary='Core cost excludes Pillar/nonconv/pack/selection/warp; no AP, speed or SNN claim; shared GPU allowed.'))
    _,stream=loader(rows[:a.frames],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured=[]
            hook=base.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:expected,base_state,_=base.step(ego,base_state,meta)
            finally:hook.remove()
            entry=dict(frame=meta['frame'],supervised=bool(meta['supervised']),arms={});preds={}
            for name,model in models.items():
                old_state=states[name]
                (pred,states[name],info),cost=measured_step(model,captured[0],old_state,meta)
                preds[name]=pred
                if name=='period1':assert all(torch.equal(pred[k],expected[k]) for k in pred)
                if info['late_refresh'] is None:
                    assert states[name]['anchor']['value'] is old_state['anchor']['value']
                    assert states[name]['anchor']['timestamp_us']==old_state['anchor']['timestamp_us']
                entry['arms'][name]=dict(core_conv_flops=cost,**info)
            assert all(torch.allclose(preds['patch'][k],preds['reference'][k],atol=1e-4,rtol=1e-4) for k in preds['patch'])
            for key in ('value','coarse'):assert torch.equal(states['patch'][key],states['reference'][key])
            assert torch.equal(states['patch']['anchor']['value'],states['reference']['anchor']['value'])
            assert states['patch']['anchor']['timestamp_us']==states['reference']['anchor']['timestamp_us']
            assert entry['arms']['patch']['core_conv_flops']<=entry['arms']['reference']['core_conv_flops']
            entry['prediction_max_error']=max(float((preds['patch'][k]-preds['reference'][k]).abs().max()) for k in preds['patch'])
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='train_geometry',scans=len(records),total=a.frames))
    assert len(records)==a.frames
    dense=sum(r['arms']['period1']['core_conv_flops'] for r in records)
    active=[r['arms']['patch']['repair'] for r in records if r['arms']['patch']['repair'] and r['arms']['patch']['repair']['active']]
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(records),split='train',checkpoint_sha256=digest(Path(a.checkpoint)),
        core_ratios={name:sum(r['arms'][name]['core_conv_flops'] for r in records)/dense for name in models},
        patch_frames=len(active),fallback_frames=sum(s['fallback'] for s in active),
        mean_patch_write_fraction=sum(s['actual_write_fraction'] for s in active)/max(1,len(active)),
        cache_bytes=records[-1]['arms']['patch']['cache_bytes'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        period1_dense_exact_all=True,same_mask_parity_all=True,anchor_never_overwritten_on_reuse=True,
        prediction_max_error=max(r['prediction_max_error'] for r in records),ap_evaluated=False,timing_valid=False,
        caveat='Training geometry gate only. Peak GPU includes all controls/shared base; persistent cache bytes exclude temporary packed tensors.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--frames',type=int,default=256);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 32<=a.frames<=256:p.error('32 to256 training frames')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h engineering budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<10000:raise RuntimeError('At least10GB free required')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.check_int_anchor_patch']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

"""Engineering only: frozen full refresh and same-mask local cache parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import torch
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_cache import CachedPyramidDetector


def measured_step(model, x, state, meta, requests=None):
    macs = [0]
    def count(module, inputs, output):
        if isinstance(module,torch.nn.ConvTranspose2d):
            macs[0] += inputs[0].numel()*(module.out_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        else:
            macs[0] += output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
    hooks=[m.register_forward_hook(count) for m in model.core.modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))]
    try:result=model.step_spatial(x,state,meta,requests=requests)
    finally:
        for hook in hooks:hook.remove()
    return result,2*macs[0]


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    a.mode='gru';a.stage='pilot';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    model=PyramidStateDetector(teacher,cfg).cuda().eval()
    saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    assert saved['contract']['version']==model.version
    model.load_state_dict(saved['model'],strict=True)
    full=CachedPyramidDetector(model,fraction=1)
    sparse=CachedPyramidDetector(model,fraction=.1,tile=4,allow_fallback=False)
    masked=CachedPyramidDetector(model,fraction=.1,tile=4,allow_fallback=False,execution='masked_dense')
    base_state=full_state=partial_state=None;records=[]
    _,stream=loader(rows[:8],cfg,2,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];captured=[]
            hook=model.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:expected,base_state,_=model.step(ego,base_state,meta)
            finally:hook.remove()
            x=captured[0]
            (actual,full_state,_),full_cost=measured_step(full,x,full_state,meta)
            assert all(torch.equal(expected[k],actual[k]) for k in expected)
            assert all(torch.equal(base_state[k],full_state[k]) for k in model.state_keys)
            (local,next_state,info),sparse_cost=measured_step(sparse,x,partial_state,meta)
            (reference,ref_state,ref_info),masked_cost=measured_step(masked,x,partial_state,meta,sparse.last_requests)
            assert all(torch.allclose(local[k],reference[k],atol=1e-4,rtol=1e-4) for k in local)
            assert all(torch.allclose(next_state[k],ref_state[k],atol=1e-4,rtol=1e-4) for k in model.state_keys)
            assert [s['actual_write_fraction'] for s in info['cache_stages']]==[s['actual_write_fraction'] for s in ref_info['cache_stages']]
            records.append(dict(frame=meta['frame'],reset=info['reset'],full_refresh_exact=True,
                same_mask_prediction_max_error=max(float((local[k]-reference[k]).abs().max()) for k in local),
                sparse_core_flops=sparse_cost,masked_dense_core_flops=masked_cost,full_core_flops=full_cost,
                cache_bytes=info['cache_bytes'],stages=info['cache_stages']))
            partial_state=next_state
    report=dict(status='PASSED',checkpoint=a.checkpoint,checkpoint_sha256=digest(Path(a.checkpoint)),records=records,
        scope='Eight real frames engineering only; no full-validation AP or latency claim',
        limitations=['Pillar/selector/warp/deblocks/readout/heads remain dense; this is stage-cache execution, not whole-network caching.',
                     'Convolution FLOPs include executed halos and duplicated crop work; exclude nonconvolution overhead.',
                     'Partial mode matches same-mask dense intervention, not ordinary full-update predictions.',
                     'Per-frame nearest warp may accumulate resampling drift; per-cell age guard is not a proof of AP preservation.'])
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h engineering budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('Insufficient free GPU memory')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.check_int_pyramid_cache']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,cwd=str(Path.cwd()),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

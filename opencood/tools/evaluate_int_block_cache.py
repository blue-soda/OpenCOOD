"""Registered frozen block-cache full-stream AP and executed convolution work."""
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
from opencood.tools.run_int_cache_repair import setup, evaluate
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


class MeasuredCache:
    def __init__(self, model):
        self.model=model;self.core=0;self.pillar=0;self.records=[];self.handles=[]
        def core_count(module,inputs,output):
            if isinstance(module,torch.nn.ConvTranspose2d):
                n=inputs[0].numel()*module.out_channels//module.groups
            else:n=output.numel()*module.in_channels//module.groups
            self.core+=2*n*module.kernel_size[0]*module.kernel_size[1]
        def pillar_count(module,inputs,output):
            self.pillar+=2*output.numel()*module.in_features
        for module in model.core.modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                self.handles.append(module.register_forward_hook(core_count))
        for module in model.pillar_vfe.modules():
            if isinstance(module,torch.nn.Linear):self.handles.append(module.register_forward_hook(pillar_count))

    def step(self,ego,state,meta):
        self.core=self.pillar=0
        pred,state,info=self.model.step(ego,state,meta)
        # Common evaluator expects a constant numeric schema, including resets.
        info['cache_stages']=[dict(actual_write_fraction=s['actual_write_fraction'],
            executed_input_area_ratio=s['executed_input_area_ratio'],fallback=float(s['fallback']),
            mandatory_fraction=s['mandatory_fraction']) for s in info['cache_stages']]
        info.update(core_conv_flops=self.core,pillar_linear_flops=self.pillar)
        self.records.append(dict(core=self.core,pillar=self.pillar,state_bytes=info['state_bytes'],cache_bytes=info['cache_bytes']))
        return pred,state,info

    def close(self):
        for hook in self.handles:hook.remove()


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    gate=json.loads(Path(a.engineering).read_text())
    assert gate['status']=='PASSED' and gate['granularity']=='block'
    assert gate['checkpoint_sha256']==digest(Path(a.checkpoint))
    assert len(gate['records'])>=32 and all(x['full_refresh_exact'] for x in gate['records'])
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,
        max_age=.25,oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    assert saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    model=BlockCachedPyramidDetector(base,fraction=.1,tile=4,max_cache_age=.5,execution='coalesced')
    write_json(out/'policy.json',dict(checkpoint_sha256=gate['checkpoint_sha256'],fraction=.1,tile=4,max_cache_age=.5,
        execution='coalesced',cache_units=15,trained=False,calibrated=False,
        selection='Per-block input relative change ranking plus mandatory coverage/age; fixed before full validation',
        caveat='Engineering used validation prefix; full validation is exploratory, not independent test or final acceptance'))
    measured=MeasuredCache(model);a.stage='full'
    write_json(out/'status.json',dict(status='RUNNING',phase='full_validation'))
    try:report=evaluate(a,teacher,cfg,rows,out,measured)
    finally:measured.close()
    n=len(measured.records);dense=gate['records'][0]['full_core_flops']
    report.update(scope='Frozen R051 per-block cache full validation',checkpoint_sha256=gate['checkpoint_sha256'],
        timing_valid=False,core_conv_flops_mean=sum(x['core'] for x in measured.records)/n,
        pillar_linear_flops_mean=sum(x['pillar'] for x in measured.records)/n,dense_core_conv_flops=dense,
        core_conv_ratio=sum(x['core'] for x in measured.records)/(n*dense),
        state_bytes=measured.records[-1]['state_bytes'],cache_bytes=measured.records[-1]['cache_bytes'])
    for key in list(report):
        if key.startswith('model_step_ms'):report[key]=None
    report['caveats']=['Full validation used for exploratory mechanism assessment, not independent testing.',
        'Core Conv2d/ConvTranspose2d and Pillar Linear work measured from executed shapes, including crop halos and dense fallbacks.',
        'Warp, selector, packing, normalization, activations, scatter and IO are not included in convolution/linear FLOPs.',
        'Pillar/15 selectors and warps/deblocks/readout/heads remain dense; added intermediate caches are fully counted.',
        'Instrumented timing is invalid for speed claims; partial updates are a new policy, not ordinary dense predictions.']
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','engineering'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',required=True,type=int);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('2h frozen evaluation budget')));signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('Insufficient free GPU memory')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.evaluate_int_block_cache']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

"""Frozen strict-anchor operator accounting; no AP, optimizer or timing."""
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
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING'))
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    source=Path(a.source);gold_report=json.loads(source.read_text())
    assert gold_report['status']=='COMPLETED' and gold_report['scans']==256 and gold_report['split']=='train'
    assert gold_report['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    model=AnchoredLatePyramidDetector(base);state=base_state=None;records=[]
    modules={n:m for n,m in model.core.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))}
    description={n:dict(type=type(m).__name__,in_channels=m.in_channels,out_channels=m.out_channels,
        kernel=list(m.kernel_size),stride=list(m.stride),groups=m.groups,
        parameters=sum(p.numel() for p in m.parameters())) for n,m in modules.items()}
    write_json(out/'contract.json',dict(split='train',frames=256,source_sha256=digest(source),
        checkpoint_sha256=digest(Path(a.checkpoint)),policy='Frozen strict period2 coverage/max_age=.25',
        formula='Conv2d: 2*output.numel*Cin/groups*kernel_area; transpose: 2*input.numel*Cout/groups*kernel_area.',
        exclusions='Pillar, linear layers, BN/ReLU, warp, scheduling and memory traffic; no timing or energy claim.',
        validation='Each frame sum equals independent measured_step counter and archived R083 cost; first32 replay without profile hooks has exact predictions and states.',
        ap_evaluated=False,trained=False,modules=description))
    _,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            calls={};hooks=[]
            def counter(name):
                def collect(m,inputs,output):
                    size=inputs[0].numel() if isinstance(m,torch.nn.ConvTranspose2d) else output.numel()
                    channels=m.out_channels if isinstance(m,torch.nn.ConvTranspose2d) else m.in_channels
                    calls.setdefault(name,[]).append(dict(input=list(inputs[0].shape),output=list(output.shape),
                        flops=2*size*(channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]))
                return collect
            for name,module in modules.items():hooks.append(module.register_forward_hook(counter(name)))
            previous=state
            try:(pred,state,info),cost=measured_step(model,x,previous,meta)
            finally:
                for hook in hooks:hook.remove()
            assert sum(c['flops'] for values in calls.values() for c in values)==cost
            expected=gold[index]['arms']['period2_strict']
            assert meta['frame']==gold[index]['frame'] and cost==expected['core_conv_flops'] and info['late_refresh']==expected['late_refresh']
            if index<32:
                ref,refstate,refinfo=model.step_spatial(x,previous,meta)
                assert all(torch.equal(pred[k],ref[k]) for k in pred)
                assert all(torch.equal(state[k],refstate[k]) for k in ('value','coarse'))
                assert torch.equal(state['anchor']['value'],refstate['anchor']['value'])
                assert refinfo['late_refresh']==info['late_refresh']
            entry=dict(frame=meta['frame'],refresh=info['late_refresh'],core_flops=cost,operators=calls)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records)))
    assert len(records)==256
    total=sum(r['core_flops'] for r in records);operators={};stages={}
    for name,desc in description.items():
        calls=[c for r in records for c in r['operators'].get(name,[])]
        flops=sum(c['flops'] for c in calls);operators[name]=dict(calls=len(calls),flops=flops,fraction=flops/total,**desc)
        stage=name.split('.')[0];stages[stage]=stages.get(stage,0)+flops
    assert sum(stages.values())==total
    write_json(out/'report.json',dict(status='COMPLETED',scans=256,split='train',checkpoint_sha256=digest(Path(a.checkpoint)),
        total_core_flops=total,operators=operators,stages={k:dict(flops=v,fraction=v/total) for k,v in stages.items()},
        refresh_frames=sum(r['refresh'] is not None for r in records),counter_matches_source_all=True,
        no_profile_hook_prediction_state_exact_frames=32,ap_evaluated=False,timing_valid=False,
        caveat='Operator coverage only. Converting activations does not itself remove these convolutions; complete removal is an unattainable upper bound, not a measured saving.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h operator audit budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.profile_int_pyramid_budget']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

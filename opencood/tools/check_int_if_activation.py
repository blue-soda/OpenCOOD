"""Train activation calibration and single-site IF/count engineering gate."""
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
from opencood.models.sub_modules.int_if_activation import coded_anchor
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


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
    assert source_report['status']=='COMPLETED' and source_report['split']=='train' and source_report['scans']==256
    assert source_report['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),source_sha256=digest(source),
        split='train',calibration_frames=256,engineering_frames=32,site='deep_update.residual[1]',
        scale='Per-channel0.999 quantile of64 deterministic rotating spatial samples per actual strict refresh, min1e-6.',
        sampling='indices=(arange(min(64,N))*N//min(64,N)+17*frame_index)%N, fixed independently of AP',
        quantum=65536,steps=[4,8,16],if_reference_steps=8,ap_evaluated=False,trained=False,
        policy='Original strict period2/.25s/coverage; replace only first residual ReLU of deep StateUpdate.',
        boundary='Activation is decoded to analog before next dense convolution. IF membrane resets per call, no cross-frame membrane, sparse synapses, speed or energy claim.'))
    strict=AnchoredLatePyramidDetector(base);base_state=state=None;samples=[];calibration=[]
    _,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'calibration_frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            observed=[]
            def collect(module,args,output):
                flat=output.detach()[0].flatten(1);n=flat.shape[1];k=min(64,n)
                indices=(torch.arange(k,device=flat.device)*n//k+17*index)%n
                observed.append(flat[:,indices].cpu())
            h=base.core.deep_update.residual[1].register_forward_hook(collect)
            try:(_,state,info),cost=measured_step(strict,x,state,meta)
            finally:h.remove()
            expected=gold[index]['arms']['period2_strict'];assert meta['frame']==gold[index]['frame']
            assert info['late_refresh']==expected['late_refresh'] and cost==expected['core_conv_flops']
            assert len(observed)==int(info['late_refresh'] is not None)
            samples.extend(observed);entry=dict(frame=meta['frame'],refresh=info['late_refresh'],sample_columns=sum(t.shape[1] for t in observed),core_flops=cost)
            calibration.append(entry);log.write(json.dumps(entry)+'\n')
            if len(calibration)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='activation_calibration',scans=len(calibration)))
    assert len(calibration)==256 and len(samples)==149
    values=torch.cat(samples,1);scales=torch.quantile(values,.999,dim=1).clamp_min(1e-6)
    torch.save(values,out/'activation_samples.pt')
    write_json(out/'scales.json',dict(scales=scales.tolist(),channels=len(scales),samples_per_channel=values.shape[1],
        quantile=.999,minimum=1e-6,sample_sha256=digest(out/'activation_samples.pt')))
    del values,samples
    models=dict(ann=AnchoredLatePyramidDetector(base),clip=coded_anchor(base,scales,'clip',8),
        count4=coded_anchor(base,scales,'count',4),count8=coded_anchor(base,scales,'count',8),
        count16=coded_anchor(base,scales,'count',16),if8=coded_anchor(base,scales,'if',8))
    states={name:None for name in models};base_state=None;records=[]
    params={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()};assert len(set(params.values()))==1
    _,stream=loader(rows[:32],cfg,2,303)
    with torch.no_grad(),(out/'engineering_frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            preds={};entry=dict(frame=meta['frame'],arms={})
            for name,model in models.items():
                codec=None if name=='ann' else model.core.deep_update.residual[1]
                calls=codec.calls if codec is not None else 0
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);preds[name]=pred
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert info['late_refresh']==gold[index]['arms']['period2_strict']['late_refresh']
                assert cost==gold[index]['arms']['period2_strict']['core_conv_flops']
                assert all(torch.equal(states[name][k],base_state[k]) for k in ('value','coarse'))
                if codec is not None:assert codec.calls-calls==int(info['late_refresh'] is not None)
                entry['arms'][name]=dict(core_flops=cost,refresh=info['late_refresh'],tensor_cache_bytes=info['cache_bytes'],
                    coding=codec.last_stats if codec is not None and info['late_refresh'] is not None else None,
                    output_relative_mse={k:float((pred[k]-preds['ann'][k]).square().mean()/(preds['ann'][k].square().mean()+1e-12)) for k in pred})
            assert all(torch.equal(preds['count8'][k],preds['if8'][k]) for k in preds['count8'])
            assert torch.equal(states['count8']['anchor']['value'],states['if8']['anchor']['value'])
            entry['if_count_exact']=True;records.append(entry);log.write(json.dumps(entry)+'\n')
    assert len(records)==32
    metrics={}
    for name in models:
        activations=[r['arms'][name]['coding'] for r in records if r['arms'][name]['coding'] is not None]
        metrics[name]=dict(mean_relative_mse={k:sum(r['arms'][name]['output_relative_mse'][k] for r in records)/32 for k in preds[name]},
            actual_core_equals_ann=True,activation_calls=len(activations),
            mean_clipped_fraction=sum(x['clipped_fraction'] for x in activations)/max(1,len(activations)),
            explicit_neuron_updates=sum(x['if_neuron_updates'] for x in activations),
            mean_emission_rate=None if not activations or activations[0]['emission_rate'] is None else sum(x['emission_rate'] for x in activations)/len(activations))
    write_json(out/'report.json',dict(status='COMPLETED',split='train',calibration_frames=256,engineering_frames=32,
        checkpoint_sha256=digest(Path(a.checkpoint)),scale_sha256=digest(out/'scales.json'),parameters=params,
        if_count_exact_all=True,metrics=metrics,tensor_cache_bytes=records[-1]['arms']['ann']['tensor_cache_bytes'],
        additional_scale_buffer_bytes=4*len(scales),persistent_membrane_bytes=0,ap_evaluated=False,timing_valid=False,
        caveat='One hybrid activation site; count-decoded dense downstream Conv unchanged. Explicit IF adds neuron operations without skipping any Conv. Calibration samples limited, no AP claim.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','strict-source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h coding gate budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.check_int_if_activation']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

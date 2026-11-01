"""Train-only two-site readout IF calibration, engineering gate and frozen AP."""
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
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


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
    source=Path(a.source);gold_report=json.loads(source.read_text())
    assert gold_report['status']=='COMPLETED' and gold_report['scans']==256 and gold_report['split']=='train'
    assert gold_report['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
    write_json(out/'contract.json',dict(split='train',frames=256,checkpoint_sha256=digest(Path(a.checkpoint)),source_sha256=digest(source),
        sites=['readout input after concatenated nonnegative deblocks','readout.layers.0.double_conv.2'],
        scale='ANN per-channel0.999 quantile of64 rotating spatial samples per site/frame; min1e-6; no AP selection',
        sampling='indices=(arange(min(64,N))*N//min(64,N)+17*frame_index)%N',
        steps=8,quantum=65536,policy='Frozen strict period2/.25s/coverage; all recurrent states identical to strict.',
        candidates=['ann','clip','count8','if8','zero_mid'],zero_control='Only middle activation zero; input remains analog.',
        engineering='All INT tests; per-frame IF/count and cache/Conv checks precede AP accumulation; first32 gate separately recorded.',
        boundary='Hybrid decoded activation codes, downstream dense Conv/BN unchanged. No sparse execution, temporal membrane, timing or energy claim.',trained=False))
    strict=AnchoredLatePyramidDetector(base);base_state=state=None;samples=dict(input=[],middle=[])
    _,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'calibration_frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            observed={}
            def collect(name,tensor):
                assert bool((tensor>=0).all());flat=tensor.detach()[0].flatten(1);n=flat.shape[1];k=min(64,n)
                indices=(torch.arange(k,device=flat.device)*n//k+17*index)%n
                observed[name]=flat[:,indices].cpu()
            h1=base.core.readout.register_forward_pre_hook(lambda m,args:collect('input',args[0]))
            h2=base.core.readout.layers[0].double_conv[2].register_forward_hook(lambda m,args,y:collect('middle',y))
            try:(_,state,info),cost=measured_step(strict,x,state,meta)
            finally:h1.remove();h2.remove()
            assert meta['frame']==gold[index]['frame'] and cost==gold[index]['arms']['period2_strict']['core_conv_flops']
            assert info['late_refresh']==gold[index]['arms']['period2_strict']['late_refresh'] and set(observed)==set(samples)
            for key,value in observed.items():samples[key].append(value)
            log.write(json.dumps(dict(frame=meta['frame'],refresh=info['late_refresh'],core_flops=cost))+'\n')
            if (index+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='calibration',scans=index+1))
    values={k:torch.cat(v,1) for k,v in samples.items()};assert all(v.shape[1]==16384 for v in values.values())
    scales={k:torch.quantile(v,.999,dim=1).clamp_min(1e-6).tolist() for k,v in values.items()}
    torch.save(values,out/'activation_samples.pt');write_json(out/'scales.json',dict(scales=scales,samples_per_channel=16384,sample_sha256=digest(out/'activation_samples.pt')))
    del samples,values
    models=dict(ann=AnchoredLatePyramidDetector(base),**{n:readout_coded_anchor(base,scales,m,8) for n,m in [('clip','clip'),('count8','count'),('if8','if'),('zero_mid','zero_mid')]})
    states={k:None for k in models};base_state=None;records=[];ap={name:{.7:dict(tp=[],fp=[],score=[],gt=0)} for name in models}
    params={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()};assert len(set(params.values()))==1
    dataset,stream=loader(rows[:256],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            preds={};entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for name,model in models.items():
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);preds[name]=pred
                assert cost==gold[index]['arms']['period2_strict']['core_conv_flops'] and info['late_refresh']==gold[index]['arms']['period2_strict']['late_refresh']
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert all(torch.equal(states[name][k],states['ann'][k]) for k in ('value','coarse'))
                assert torch.equal(states[name]['anchor']['value'],states['ann']['anchor']['value'])
                coding=None if name=='ann' else model.core.readout.coding_stats()
                if name!='ann':
                    assert model.core.readout.middle_codec.calls==index+1
                    if name!='zero_mid':assert model.core.readout.input_codec.calls==index+1
                entry['arms'][name]=dict(core_flops=cost,refresh=info['late_refresh'],coding=coding,
                    relative_mse={k:float((pred[k]-preds['ann'][k]).square().mean()/(preds['ann'][k].square().mean()+1e-12)) for k in pred})
            assert all(torch.equal(preds['count8'][k],preds['if8'][k]) for k in preds['count8'])
            entry['if_count_exact']=True;entry['all_cache_states_equal_ann']=True
            # Evaluate only after the per-frame gate; first32 engineering is separately recorded.
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred));eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],.7)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if index==31:write_json(out/'engineering.json',dict(status='PASSED',frames=32,if_count_exact=True,all_cache_states_equal_ann=True,all_core_cost_equal_ann=True))
            if (index+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='frozen_diagnostic',scans=index+1))
    assert len(records)==256
    write_json(out/'ap_records.json',ap)
    metrics={}
    for name in models:
        coding=[c for r in records for c in (r['arms'][name]['coding'] or {}).values() if c is not None]
        metrics[name]=dict(ap70=eval_utils.calculate_ap(ap[name],.7)[0],core_flops=sum(r['arms'][name]['core_flops'] for r in records),
            explicit_neuron_updates=sum(c['if_neuron_updates'] for c in coding),
            mean_relative_mse={k:sum(r['arms'][name]['relative_mse'][k] for r in records)/256 for k in preds[name]},
            mean_clipped_fraction=sum(c['clipped_fraction'] for c in coding)/max(1,len(coding)),
            emission_rate=None if name not in ('count8','if8') else sum(c['spike_count'] for c in coding)/sum(c['steps']*c['elements'] for c in coding))
    assert abs(metrics['ann']['ap70']-gold_report['metrics']['period2_strict']['ap70'])<1e-12
    write_json(out/'report.json',dict(status='COMPLETED',split='train',scans=256,labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),scale_sha256=digest(out/'scales.json'),parameters=params,metrics=metrics,
        if_count_exact_all=True,all_cache_states_equal_ann=True,all_core_cost_equal_ann=True,scale_buffer_bytes=2560,
        persistent_membrane_bytes=0,timing_valid=False,caveat='Train-prefix calibration and diagnostic only; fixed T8, no AP tuning. Dense convolutions unchanged; no sparse execution or SNN benefit claim.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','checkpoint','source'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1h readout coding budget')));signal.alarm(3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.diagnose_int_readout_if']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

"""Frozen amplitude engineering gate; fixed calibration, no AP or optimizer."""
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
from opencood.models.sub_modules.int_readout_amplitude import ReadoutAmplitudeAnchor,integer_delivery_statistics
from opencood.models.sub_modules.int_spatial_spike import deliveries3x3
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.pilot_int_readout_spike import same_ann_cache
from opencood.tools.pilot_int_spatial_spike import support_histogram
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def make_models(base,scale):
    models=dict(strict=AnchoredLatePyramidDetector(base),ann=ReadoutAmplitudeAnchor(base,'ann',scale),
        count=ReadoutAmplitudeAnchor(base,'count',scale))
    for mode in ('residual','ema'):
        for dtype in (torch.float32,torch.float16):
            models[mode+('_fp32' if dtype==torch.float32 else '_fp16')]=ReadoutAmplitudeAnchor(base,mode,scale,dtype)
        models[mode+'_reset']=ReadoutAmplitudeAnchor(base,mode,scale,torch.float16,True)
    for model in models.values():
        for p in model.parameters():p.requires_grad_(False)
    return models


def clear_diagnostics(models):
    for name,model in models.items():
        if name!='strict':
            model.last_input=model.last_output=model.last_common_input=None;model.activation_memory.last_counts=None


def frozen_frame(models,states,x,meta):
    record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
    for name,model in models.items():
        old=None if states[name] is None else states[name].get('readout_membrane')
        old_value=None if old is None else old['value'].clone()
        (pred,states[name],info),cost=measured_step(model,x,states[name],meta);predictions[name]=pred
        assert all(bool(torch.isfinite(v).all()) for v in pred.values())
        if old is not None:assert torch.equal(old['value'],old_value)
        extra=0 if name in ('strict','ann','count') else (25804800 if name.endswith('_fp32') else 12902400)
        assert info['cache_bytes']==22579200+extra
        entry=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=info.get('readout_activation'))
        if name!='strict':
            assert cost==record['arms']['strict']['core_conv_flops'] and info['late_refresh']==record['arms']['strict']['refresh']
            same_ann_cache(states[name],states['strict'])
            assert torch.equal(model.last_input,models['ann'].last_input) and torch.equal(model.last_common_input,models['ann'].last_common_input)
            source=model.last_input;output=model.last_output
            entry.update(input_elements=source.numel(),input_support_4_6_9=support_histogram(source),output_support_4_6_9=support_histogram(output),
                input_sparse_macs=deliveries3x3(source!=0,256),output_sparse_macs=deliveries3x3(output!=0,256),
                input_mse=float((output-source).square().mean()),input_max_error=float((output-source).abs().max()),
                prediction_max_error=max(float((v-predictions['strict'][k]).abs().max()) for k,v in pred.items()))
            if model.activation_memory.last_counts is not None:entry['events']=integer_delivery_statistics(model.activation_memory.last_counts)
            if name in ('ann','ema_reset'):assert all(torch.equal(v,predictions['strict'][k]) for k,v in pred.items())
            if name=='residual_reset':
                assert torch.equal(model.last_output,models['count'].last_output)
                assert all(torch.equal(v,predictions['count'][k]) for k,v in pred.items())
            if name in ('ema_fp16','residual_fp16'):
                ref=name.replace('_fp16','_fp32');delta=states[name]['readout_membrane']['value'].float()-states[ref]['readout_membrane']['value']
                entry['fp16_vs_fp32']=dict(state_max_error=float(delta.abs().max()),state_mse=float(delta.square().mean()),
                    output_support_disagreements=int(torch.count_nonzero((output!=0)!=(models[ref].last_output!=0))),
                    output_max_error=float((output-models[ref].last_output).abs().max()),
                    prediction_max_error=max(float((v-predictions[ref][k]).abs().max()) for k,v in pred.items()))
                if name=='residual_fp16':entry['fp16_vs_fp32']['count_disagreements']=int(torch.count_nonzero(model.activation_memory.last_counts!=models[ref].activation_memory.last_counts))
        record['arms'][name]=entry
    clear_diagnostics(models)
    return record,predictions


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    reference=Path(a.reference);verified=json.loads(Path(a.reference_verification).read_text())
    assert verified['status']=='VERIFIED'
    for n in ('report.json','contract.json','calibration.json','frames.jsonl'):assert digest(reference/n)==verified['source_sha256'][n]
    prior=json.loads((reference/'report.json').read_text());cal=json.loads((reference/'calibration.json').read_text());pc=json.loads((reference/'contract.json').read_text())
    assert prior['status']=='COMPLETED' and prior['checkpoint_sha256']==cal['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(line) for line in (reference/'frames.jsonl').read_text().splitlines()]
    data=json.loads(Path(a.data_audit).read_text());assert data['status']=='VERIFIED' and data['frames']==288
    for row in data['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    selected=rows[512:544];assert [r['frame'] for r in selected]==pc['evaluation_frames']
    assert [r['frame'] for r in rows[:256]]==cal['frames']
    assert sorted({r[k] for r in rows[:256]+selected for k in ('pcd','label') if r[k]})==sorted(r['relocated_path'] for r in data['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);models=make_models(base,cal['scale'])
    models={n:m.cuda().eval() for n,m in models.items()};hashes={n:model_digest(m) for n,m in models.items()}
    assert all(sum(p.numel() for p in m.parameters())==32756052 and not any(p.requires_grad for p in m.parameters()) for m in models.values())
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),reference_report_sha256=digest(reference/'report.json'),reference_verification_sha256=digest(Path(a.reference_verification)),
        calibration_sha256=digest(reference/'calibration.json'),data_audit_sha256=digest(Path(a.data_audit)),scale=cal['scale'],quantum_rule='q=s/8, fixed before AP',
        evaluation_frames=[r['frame'] for r in selected],split='train',seed=303,parameters=32756052,trainable_parameters=0,tau=.2,
        storage='FP16 persistent state / FP32 dynamics, paired FP32 controls',trained=False,ap_evaluated=False,timing_valid=False,
        cost_scope='Core Conv FLOPs; exact integer unary AC and bitplane additions; excludes scale/shift/routing/neuron/warp/storage costs, not hardware measurement')
    write_json(out/'contract.json',contract);states={n:None for n in models};base_state=None;records=[]
    _,stream=loader(selected,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            record,_=frozen_frame(models,states,x,meta)
            assert record['frame']==gold[i]['frame'] and record['supervised']==gold[i]['supervised']
            assert record['arms']['strict']==gold[i]['arms']['strict']
            records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%8==0:write_json(out/'status.json',dict(status='RUNNING',phase='engineering',scans=i+1,total=32))
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=32756052,core_conv_flops_mean=sum(v['arms'][n]['core_conv_flops'] for v in records)/32,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        ap50=None,ap70=None,model_step_ms_mean=None,reason='32frame engineering only; no AP or valid latency') for n in models}
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],scans=32,labels=sum(v['supervised'] for v in records),metrics=metrics,
        all_weights_bn_scale_unchanged=True,common_input_and_ann_cache_exact=True,ann_and_ema_reset_exact=True,residual_reset_count_exact=True,
        old_state_unmutated=True,residual_and_amplitude_bounds_checked=True,trained=False,ap_evaluated=False,timing_valid=False,
        process_cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,
        boundary='Frozen amplitude representation engineering; no surrogate gradients, no AP/energy/speed claim. Integer multiplicity and bitplanes have distinct cost assumptions.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','reference','reference-verification','data-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min amplitude gate budget')));signal.alarm(2700)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free, process cap50%; shared correctness only')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.check_int_readout_amplitude']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

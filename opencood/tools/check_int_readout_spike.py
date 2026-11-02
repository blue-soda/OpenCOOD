"""Train-only scale calibration and 32-frame readout FP16 storage engineering gate."""
import argparse
import copy
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
from opencood.models.sub_modules.int_spatial_spike import deliveries3x3
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.models.sub_modules.int_temporal_feature import detach_temporal_state
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def gradient_probe(model,frames):
    before=model_digest(model);result={}
    for branch in ('unscaled_connected','connected','detach','full_reset'):
        model.zero_grad(set_to_none=True);state=None
        for i,(source,source_meta) in enumerate(frames):
            x=source.cuda().detach().requires_grad_(i==0);meta=copy.deepcopy(source_meta)
            if i==1:
                if branch=='detach':state=detach_temporal_state(state)
                if branch=='full_reset':meta['reset_before']=True
            pred,state,_=model.step_spatial(x,state,meta,detach_state=False)
            if i==0:first=x;membrane=state['readout_membrane']['value'];membrane.retain_grad()
        loss=pred['psm'].square().mean()+pred['rm'].square().mean();assert bool(torch.isfinite(loss))
        loss_scale=1 if branch=='unscaled_connected' else 4096
        (loss*loss_scale).backward()
        norm=lambda p:0. if p.grad is None else float(p.grad.float().abs().sum())
        assert first.grad is None or bool(torch.isfinite(first.grad).all())
        assert membrane.grad is None or bool(torch.isfinite(membrane.grad).all())
        parameters=dict(model.named_parameters())
        norms={n:norm(parameters[n])/loss_scale for n in model.trainable_names}
        entry=dict(loss=float(loss.detach()),loss_scale=loss_scale,first_input_gradient_l1=norm(first)/loss_scale,
            first_membrane_gradient_l1=norm(membrane)/loss_scale,trainable_gradient_l1=norms)
        assert all(bool(torch.isfinite(parameters[n].grad).all()) for n in model.trainable_names)
        assert all(p.grad is None for n,p in model.named_parameters() if n not in model.trainable_names)
        if branch!='unscaled_connected':
            assert all(v>0 for v in norms.values())
            if branch=='connected':assert entry['first_input_gradient_l1']>0 and entry['first_membrane_gradient_l1']>0
            else:assert entry['first_input_gradient_l1']==entry['first_membrane_gradient_l1']==0
        def check(v):
            if torch.is_tensor(v):assert v.grad_fn is None and not v.requires_grad
            elif isinstance(v,dict):
                for child in v.values():check(child)
        check(detach_temporal_state(state));result[branch]=entry
    model.zero_grad(set_to_none=True);assert model_digest(model)==before
    model.last_input=model.last_output=model.last_common_input=None
    return dict(branches=result,weights_and_bn_unchanged=True,all_state_detach_verified=True,
        frames=[m['frame'] for _,m in frames],loss='Head-square mean; fixed4096 loss scaling for topology gate; unscaled connected diagnostic retained; no optimizer')


def support_buckets(values):
    active=values!=0;h,w=values.shape[-2:]
    y=torch.full((h,),3,device=values.device,dtype=torch.int64);y[0]-=1;y[-1]-=1
    x=torch.full((w,),3,device=values.device,dtype=torch.int64);x[0]-=1;x[-1]-=1
    fanout=y[:,None]*x[None,:]
    return {str(k):int((active*(fanout==k)).sum()) for k in (4,6,9)}


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed before data replay')
    data_audit=Path(a.data_audit);data=json.loads(data_audit.read_text())
    assert data['status']=='VERIFIED' and data['frames']==288
    for row in data['files']:
        assert digest(Path(row['relocated_path']))==row['sha256'],row['relocated_path']
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    training,evaluation=rows[:256],rows[512:544]
    assert not set(r['frame'] for r in training)&set(r['frame'] for r in evaluation)
    assert sorted({r[k] for r in training+evaluation for k in ('pcd','label') if r[k]})==sorted(r['relocated_path'] for r in data['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    calibrator=ReadoutSpikeAnchor(base,'ann',torch.ones(256)).cuda().eval();before=model_digest(calibrator)
    square=torch.zeros(256,dtype=torch.float64);count=torch.zeros(256,dtype=torch.int64);state=base_state=None
    _,stream=loader(training,cfg,2,303)
    write_json(out/'status.json',dict(status='RUNNING',phase='calibration'))
    with torch.no_grad(),(out/'calibration_frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            _,state,_=calibrator.step_spatial(x,state,meta);values=calibrator.last_input
            assert bool(torch.isfinite(values).all()) and bool((values>=0).all())
            sq=values.double().square().sum((0,2,3)).cpu();ct=(values>0).sum((0,2,3)).cpu()
            square+=sq;count+=ct;log.write(json.dumps(dict(frame=meta['frame'],positive_square_sum=sq.tolist(),positive_count=ct.tolist()))+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='calibration',scans=i+1,total=256))
    assert i+1==256 and model_digest(calibrator)==before
    scale=torch.where(count>0,(square/count.clamp_min(1)).sqrt().clamp_min(1e-6),torch.ones_like(square)).float()
    write_json(out/'calibration.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),frames=[r['frame'] for r in training],
        positive_square_sum=square.tolist(),positive_count=count.tolist(),scale=scale.tolist(),rule='Positive RMS per channel; silent=1; floor1e-6; float32 scale'))
    del calibrator;state=base_state=None
    models={'strict':AnchoredLatePyramidDetector(base),'ann':ReadoutSpikeAnchor(base,'ann',scale).cuda().eval()}
    for mode in ('lif','leaky'):
        for dtype in (torch.float32,torch.float16):
            name=mode+('_fp32' if dtype==torch.float32 else '_fp16')
            models[name]=ReadoutSpikeAnchor(base,mode,scale,dtype).cuda().eval()
        models[mode+'_fp16_reset']=ReadoutSpikeAnchor(base,mode,scale,torch.float16,True).cuda().eval()
    hashes={n:model_digest(m) for n,m in models.items()}
    assert all(sum(p.numel() for p in m.parameters())==32756052 for m in models.values())
    assert all(sum(p.numel() for p in m.parameters() if p.requires_grad)==1475072 for n,m in models.items() if n!='strict')
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),data_audit_sha256=digest(data_audit),calibration_sha256=digest(out/'calibration.json'),
        calibration_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],target='readout.layers.0.double_conv.3',
        parameters=32756052,eligible_trainable_parameters=1475072,scale_buffer_bytes=1024,tau=.2,threshold=1.,within_frame_steps=1,
        split='train',seed=303,trained=False,ap_evaluated=False,timing_valid=False,
        operation_scope='Actual core dense Conv FLOPs. Ideal target AC for binary lif; sparse MAC for continuous ann/leaky. Exact valid spatial fanout; excludes routing, warp, neuron and memory costs.')
    write_json(out/'contract.json',contract)
    states={n:None for n in models};base_state=None;records=[];probe=[];_,stream=loader(evaluation,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            if len(probe)<4:
                if probe and (meta['segment']!=probe[-1][1]['segment'] or not 0<int(meta['timestamp_us'])-int(probe[-1][1]['timestamp_us'])<=250000 or meta.get('reset_before',False)):probe=[]
                probe.append((x.detach().cpu(),copy.deepcopy(meta)))
            record=dict(frame=meta['frame'],supervised=meta['supervised'],timestamp_us=meta['timestamp_us'],segment=meta['segment'],reset_before=bool(meta.get('reset_before',False)),arms={});predictions={}
            for name,model in models.items():
                old=None if states[name] is None else states[name].get('readout_membrane')
                old_value=None if old is None else old['value'].clone()
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);predictions[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                if name!='strict':assert cost==record['arms']['strict']['core_conv_flops'] and info['late_refresh']==record['arms']['strict']['refresh']
                expected_bytes=22579200+(0 if name in ('strict','ann') else (25804800 if name.endswith('fp32') else 12902400))
                assert info['cache_bytes']==expected_bytes
                if old is not None:assert torch.equal(old_value,old['value'])
                entry=dict(core_conv_flops=cost,cache_bytes=expected_bytes,refresh=info['late_refresh'],memory=info.get('readout_activation'))
                if name!='strict':
                    assert torch.equal(model.last_input,models['ann'].last_input)
                    assert torch.equal(model.last_common_input,models['ann'].last_common_input)
                    for k in ('value','coarse'):assert torch.equal(states[name][k],states['strict'][k])
                    assert torch.equal(states[name]['anchor']['value'],states['strict']['anchor']['value'])
                    source=model.last_input;output=model.last_output
                    entry.update(input_support_buckets=support_buckets(source),output_support_buckets=support_buckets(output),input_nonzero=int(torch.count_nonzero(source)),output_nonzero=int(torch.count_nonzero(output)),input_elements=source.numel(),
                        ideal_input_sparse_macs=deliveries3x3(source!=0,256),ideal_output_deliveries=deliveries3x3(output!=0,256),
                        target_dense_macs=source.numel()*256*9,
                        target_input_mse=float((output-source).square().mean()),target_input_abs_max_error=float((output-source).abs().max()),
                        prediction_max_error=max(float((v-predictions['strict'][k]).abs().max()) for k,v in pred.items()))
                    if name=='ann':
                        assert all(torch.equal(v,predictions['strict'][k]) for k,v in pred.items())
                        assert all(torch.equal(states[name][k],states['strict'][k]) for k in ('value','coarse'))
                        assert torch.equal(states[name]['anchor']['value'],states['strict']['anchor']['value'])
                    if name.startswith('lif'):assert bool(((output==0)|(output==model.activation_memory.scale)).all())
                if name in ('lif_fp16','leaky_fp16'):
                    counterpart=name.replace('fp16','fp32');original=states[counterpart]['readout_membrane']['value']
                    delta=states[name]['readout_membrane']['value'].float()-original
                    entry['fp16_vs_fp32']=dict(membrane_max_error=float(delta.abs().max()),membrane_mse=float(delta.square().mean()),
                        prediction_max_error=max(float((v-predictions[counterpart][k]).abs().max()) for k,v in pred.items()))
                record['arms'][name]=entry
            records.append(record);log.write(json.dumps(record)+'\n')
    assert len(records)==32 and len(probe)==4
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'status.json',dict(status='RUNNING',phase='gradient_probe'))
    gradients={n:gradient_probe(models[n],probe) for n in ('lif_fp32','leaky_fp32','lif_fp16','leaky_fp16')};write_json(out/'gradients.json',gradients)
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=32756052,core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        model_step_ms_mean=None,ap50=None,ap70=None,reason='Engineering replay and gradient fixture; no AP or valid timing') for n in models}
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],calibration_scans=256,evaluation_scans=32,
        evaluation_labels=sum(r['supervised'] for r in records),metrics=metrics,ann_predictions_and_states_exact=True,common_input_exact=True,
        process_cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_memory_scope='Process peak over resident eight models, calibration and gradient fixture; not per-arm inference peak',
        no_old_state_mutation=True,weights_and_bn_unchanged=True,gradient_gate_passed=True,trained=False,ap_evaluated=False,timing_valid=False,
        boundary='Frozen scale, one real scan per update, own recurrent state per arm. Readout membrane uses FP32 dynamics and optional FP16 persistence; actual Conv still dense. No AP or hardware speed/energy improvement claimed.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','data-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min engineering budget')));signal.alarm(2700)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free; shared correctness only, 50% process cap')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.check_int_readout_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

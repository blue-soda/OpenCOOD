"""Train-only scale calibration and 32-frame spatial activation engineering gate."""
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
from opencood.models.sub_modules.int_spatial_spike import SpatialSpikeAnchor,deliveries3x3
from opencood.models.sub_modules.int_temporal_feature import detach_temporal_state
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def gradient_probe(model,frames):
    before=model_digest(model);result={}
    for branch in ('connected','detach','full_reset'):
        model.zero_grad(set_to_none=True);state=None
        for i,(source,source_meta) in enumerate(frames):
            x=source.cuda().detach().requires_grad_(i==0);meta=copy.deepcopy(source_meta)
            if i==1:
                if branch=='detach':state=detach_temporal_state(state)
                if branch=='full_reset':meta['reset_before']=True
            pred,state,_=model.step_spatial(x,state,meta,detach_state=False)
            if i==0:first=x;membrane=state['spatial_membrane']['value'];membrane.retain_grad()
        loss=pred['psm'].square().mean()+pred['rm'].square().mean();assert bool(torch.isfinite(loss));loss.backward()
        norm=lambda p:0. if p.grad is None else float(p.grad.abs().sum())
        target=model.core.coarse_encoder[3].conv2.weight
        entry=dict(loss=float(loss.detach()),first_input_gradient_l1=norm(first),first_membrane_gradient_l1=norm(membrane),target_weight_gradient_l1=norm(target))
        assert entry['target_weight_gradient_l1']>0 and bool(torch.isfinite(target.grad).all())
        assert all(p.grad is None for n,p in model.named_parameters() if n!='core.coarse_encoder.3.conv2.weight')
        if branch=='connected':assert entry['first_input_gradient_l1']>0 and entry['first_membrane_gradient_l1']>0
        else:assert entry['first_input_gradient_l1']==entry['first_membrane_gradient_l1']==0
        def check(v):
            if torch.is_tensor(v):assert v.grad_fn is None and not v.requires_grad
            elif isinstance(v,dict):
                for child in v.values():check(child)
        check(detach_temporal_state(state));result[branch]=entry
    model.zero_grad(set_to_none=True);assert model_digest(model)==before
    model.last_input=model.last_output=None
    return dict(branches=result,weights_and_bn_unchanged=True,all_state_detach_verified=True,
        frames=[m['frame'] for _,m in frames],loss='Head-square input-gradient fixture, no optimizer or detection training')


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed before data replay')
    reference=Path(a.reference);prior=json.loads((reference/'report.json').read_text());prior_contract=json.loads((reference/'contract.json').read_text())
    assert prior['status']=='COMPLETED' and prior['reference_raw_ap_equal'] and prior['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(line) for line in (reference/'frames.jsonl').read_text().splitlines()][:32]
    torch.cuda.set_per_process_memory_fraction(.65)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    training,evaluation=rows[:256],rows[512:544]
    assert [r['frame'] for r in evaluation]==prior_contract['frames'][:32]
    assert not set(r['frame'] for r in training)&set(r['frame'] for r in evaluation)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    calibrator=SpatialSpikeAnchor(base,'ann',torch.ones(256)).cuda().eval();before=model_digest(calibrator)
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
    models={'strict':AnchoredLatePyramidDetector(base),'ann':SpatialSpikeAnchor(base,'ann',scale).cuda().eval()}
    for mode in ('lif','leaky'):
        for reset in (False,True):models[mode+('_reset' if reset else '')]=SpatialSpikeAnchor(base,mode,scale,reset).cuda().eval()
    hashes={n:model_digest(m) for n,m in models.items()}
    assert all(sum(p.numel() for p in m.parameters())==32756052 for m in models.values())
    assert all(sum(p.numel() for p in m.parameters() if p.requires_grad)==589824 for n,m in models.items() if n!='strict')
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),reference_report_sha256=digest(reference/'report.json'),calibration_sha256=digest(out/'calibration.json'),
        calibration_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],target='coarse_encoder.3.conv2',
        parameters=32756052,eligible_trainable_parameters=589824,scale_buffer_bytes=1024,tau=.2,threshold=1.,within_frame_steps=1,
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
            record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
            for name,model in models.items():
                old=None if states[name] is None else states[name].get('spatial_membrane')
                old_value=None if old is None else old['value'].clone()
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);predictions[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                assert cost==gold[i]['core_conv_flops'] and info['late_refresh']==gold[i]['refresh']
                expected_bytes=22579200+(6451200 if name not in ('strict','ann') else 0)
                assert info['cache_bytes']==expected_bytes
                if old is not None:assert torch.equal(old_value,old['value'])
                entry=dict(core_conv_flops=cost,cache_bytes=expected_bytes,refresh=info['late_refresh'],memory=info.get('spatial_activation'))
                if name!='strict':
                    assert torch.equal(model.last_input,models['ann'].last_input)
                    source=model.last_input;output=model.last_output
                    entry.update(input_nonzero=int(torch.count_nonzero(source)),output_nonzero=int(torch.count_nonzero(output)),input_elements=source.numel(),
                        ideal_input_sparse_macs=deliveries3x3(source!=0,256),ideal_output_deliveries=deliveries3x3(output!=0,256),
                        target_dense_macs=source.numel()*256*9,
                        target_input_mse=float((output-source).square().mean()),target_input_abs_max_error=float((output-source).abs().max()),
                        prediction_max_error=max(float((v-predictions['strict'][k]).abs().max()) for k,v in pred.items()))
                    if name=='ann':
                        assert all(torch.equal(v,predictions['strict'][k]) for k,v in pred.items())
                        assert all(torch.equal(states[name][k],states['strict'][k]) for k in ('value','coarse'))
                        assert torch.equal(states[name]['anchor']['value'],states['strict']['anchor']['value'])
                    if name.startswith('lif'):assert bool(((output==0)|(output==model.activation_memory.scale)).all())
                record['arms'][name]=entry
            records.append(record);log.write(json.dumps(record)+'\n')
    assert len(records)==32 and len(probe)==4
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'status.json',dict(status='RUNNING',phase='gradient_probe'))
    gradients={n:gradient_probe(models[n],probe) for n in ('lif','leaky')};write_json(out/'gradients.json',gradients)
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=32756052,core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        model_step_ms_mean=None,ap50=None,ap70=None,reason='Engineering replay and gradient fixture; no AP or valid timing') for n in models}
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],calibration_scans=256,evaluation_scans=32,
        evaluation_labels=sum(r['supervised'] for r in records),metrics=metrics,ann_predictions_and_states_exact=True,common_input_exact=True,
        no_old_state_mutation=True,weights_and_bn_unchanged=True,gradient_gate_passed=True,trained=False,ap_evaluated=False,timing_valid=False,
        boundary='Frozen scale, one real scan per update, own recurrent state per arm. Added single membrane; actual Conv still dense. No AP or hardware speed/energy improvement claimed.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','reference','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min engineering budget')));signal.alarm(2700)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<20000:raise RuntimeError('Requires20GB free on one idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.check_int_spatial_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

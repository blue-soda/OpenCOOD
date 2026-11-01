"""Train32 identity/causality/cost integration gate, not AP or training."""
import argparse
import copy
import hashlib
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
from opencood.models.sub_modules.int_temporal_feature import TemporalFeatureAnchor
from opencood.models.sub_modules.int_temporal_feature import DifferentiableTemporalFeatureAnchor,detach_temporal_state
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def model_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def gradient_probe(model,frames):
    """Diagnostic head loss, no labels or optimizer; nonzero projection fixture."""
    assert len(frames)==4
    before=model_digest(model);memory=model.core.temporal_feature;results={}
    original=memory.readout.weight.detach().clone()
    with torch.no_grad():memory.readout.weight.fill_(.001/32)
    memory.record_channels=True
    for branch in ('connected','detach','explicit_reset'):
        model.zero_grad(set_to_none=True);state=None;infos=[]
        for index,(source,source_meta) in enumerate(frames):
            x=source.detach().clone().requires_grad_(index==0);meta=copy.deepcopy(source_meta)
            if index==1:
                if branch=='detach':state=detach_temporal_state(state)
                if branch=='explicit_reset':meta['reset_before']=True
            pred,state,info=model.step_spatial(x,state,meta,detach_state=False);infos.append(info['feature_memory'])
            if index==0:
                first_input=x;first_memory=state['feature_memory']['value'];first_memory.retain_grad()
        loss=pred['psm'].square().mean()+pred['rm'].square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm=lambda p:0. if p.grad is None else float(p.grad.detach().abs().sum())
        entry=dict(loss=float(loss.detach()),first_input_gradient_l1=norm(first_input),first_memory_gradient_l1=norm(first_memory),
            adapter_gradient_l1={n:norm(p) for n,p in memory.named_parameters()},channels=infos)
        assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('core.temporal_feature.'))
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in memory.parameters())
        if branch=='connected':
            assert entry['first_input_gradient_l1']>0 and entry['first_memory_gradient_l1']>0
            assert entry['adapter_gradient_l1']['input_conv.weight']>0 and entry['adapter_gradient_l1']['readout.weight']>0
        else:assert entry['first_input_gradient_l1']==0 and entry['first_memory_gradient_l1']==0
        detached=detach_temporal_state(state)
        def check(value):
            if torch.is_tensor(value):assert value.grad_fn is None and not value.requires_grad
            elif isinstance(value,dict):
                for v in value.values():check(v)
        check(detached);entry['all_nested_state_detached']=True
        results[branch]=entry
    with torch.no_grad():memory.readout.weight.copy_(original)
    model.zero_grad(set_to_none=True);memory.record_channels=False
    assert model_digest(model)==before
    return dict(branches=results,parameters_and_bn_unchanged=True,projection_fixture=.001/32,
        frames=[m['frame'] for _,m in frames],loss='Last-frame mean(psm^2)+mean(rm^2), engineering only, not detection training')


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(test.stdout)
    if test.returncode:raise RuntimeError('INT tests failed')
    torch.cuda.set_per_process_memory_fraction(.45)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    models={'strict':AnchoredLatePyramidDetector(base)}
    model_class=DifferentiableTemporalFeatureAnchor if a.gradient_gate else TemporalFeatureAnchor
    for mode in ('lif','leaky'):
        for reset in (False,True):
            models[mode+('_reset' if reset else '')]=model_class(base,mode,reset).cuda().eval()
    if a.gradient_gate:
        source=Path(a.identity_reference);old_report=json.loads(source.read_text())
        assert old_report['status']=='COMPLETED' and old_report['scans']==32 and old_report['initial_predictions_and_base_states_exact']
        assert old_report['checkpoint_sha256']==digest(Path(a.checkpoint))
        old_records=[json.loads(x) for x in (source.parent/'frames.jsonl').read_text().splitlines()]
        assert all(sum(p.numel() for p in m.parameters() if p.requires_grad)==82016 for n,m in models.items() if n!='strict')
    identity=models['lif'].core.temporal_feature.state_dict()
    assert all(all(torch.equal(v,model.core.temporal_feature.state_dict()[k]) for k,v in identity.items()) for name,model in models.items() if name!='strict')
    params={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()}
    assert all(params[n]==params['strict']+82016 for n in models if n!='strict')
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),split='train',frames=32,
        parameters=params,trained=False,ap_evaluated=False,timing_valid=False,
        gradient_gate=a.gradient_gate,identity_reference_sha256=digest(Path(a.identity_reference)) if a.gradient_gate else None,
        modes=list(models),site='post-coarse StateUpdate residual, stride4',memory_channels=32,
        tau_u=.2,tau_r=.1,threshold=1.,input_gain=.25,max_gap=.25,
        initialization='Identical adapters, zero output projection; LIF vs continuous leaky and reset controls',
        time='One physical LiDAR scan per update, true dt, both membrane and trace ego-warped; segment/gap/explicit reset',
        boundary='Dense input/readout Conv, continuous membrane+trace cache; no skipped computation or hardware benefit. Gradient gate uses frozen backbone/BN and diagnostic head loss, no optimizer.' if a.gradient_gate else 'Frozen inference integration only.'))
    states={n:None for n in models};base_state=None;records=[]
    probe=[]
    _,stream=loader(rows[:32],cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            if a.gradient_gate and len(probe)<4:
                if probe and (meta['segment']!=probe[-1][1]['segment'] or meta.get('reset_before',False) or
                              not 0<(int(meta['timestamp_us'])-int(probe[-1][1]['timestamp_us']))/1e6<=.25):probe=[]
                probe.append((x.detach().clone(),copy.deepcopy(meta)))
            record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for name,model in models.items():
                old=states[name]
                retained=None if name=='strict' or old is None else old['feature_memory']['value'].clone()
                (pred,states[name],info),cost=measured_step(model,x,old,meta);preds[name]=pred
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert all(torch.equal(v,preds['strict'][k]) for k,v in pred.items())
                assert all(torch.equal(states[name][k],states['strict'][k]) for k in ('value','coarse'))
                assert torch.equal(states[name]['anchor']['value'],states['strict']['anchor']['value'])
                if name!='strict':
                    if retained is not None:assert torch.equal(retained,old['feature_memory']['value'])
                    value=states[name]['feature_memory']['value'];h,w=value.shape[-2:]
                    extra=2*h*w*(256*32*9+32*256)
                    assert cost==record['arms']['strict']['core_conv_flops']+extra
                    assert info['cache_bytes']==record['arms']['strict']['cache_bytes']+value.numel()*value.element_size()
                    assert info['late_refresh']==record['arms']['strict']['refresh']
                    if name.endswith('_reset'):assert info['feature_memory']['reset']=='explicit'
                record['arms'][name]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=info.get('feature_memory'))
            if a.gradient_gate:assert record==old_records[len(records)]
            records.append(record);log.write(json.dumps(record)+'\n')
    assert len(records)==32
    metrics={}
    for name in models:
        metrics[name]=dict(parameters=params[name],core_conv_flops_mean=sum(r['arms'][name]['core_conv_flops'] for r in records)/32,
            cache_bytes=records[0]['arms'][name]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None,
            missing_reason='32-frame instrumented engineering gate, no AP evaluation or valid timing')
        if name!='strict':
            infos=[r['arms'][name]['memory'] for r in records]
            metrics[name].update(emission_nonzero_fraction_mean=sum(v['emission_nonzero_fraction'] for v in infos)/32,
                emission_mean=sum(v['emission_mean'] for v in infos)/32,membrane_abs_max=max(v['membrane_abs_max'] for v in infos),
                trace_abs_max=max(v['trace_abs_max'] for v in infos),aligned_history_frames=sum(v['warp'] for v in infos))
    gradient_results=None
    if a.gradient_gate:
        write_json(out/'status.json',dict(status='RUNNING',phase='four_frame_gradients'))
        gradient_results={name:gradient_probe(models[name],probe) for name in ('lif','leaky')}
        write_json(out/'gradients.json',gradient_results)
    write_json(out/'report.json',dict(status='COMPLETED',split='train',scans=32,labels=sum(r['supervised'] for r in records),
        checkpoint_sha256=digest(Path(a.checkpoint)),metrics=metrics,initial_predictions_and_base_states_exact=True,
        cost_and_cache_accounting_verified=True,adapter_weights_identical=True,old_state_not_mutated=True,
        gradient_gate=a.gradient_gate,gradient_gate_passed=True if a.gradient_gate else None,
        identity_records_equal_R102=True if a.gradient_gate else None,
        trained=False,ap_evaluated=False,timing_valid=False,
        boundary='Identity follows zero output projection, not evidence for trained accuracy or SNN advantage; inspect activity before any training.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--gradient-gate',action='store_true');p.add_argument('--identity-reference');a=p.parse_args();out=Path(a.output)
    if a.gradient_gate and not a.identity_reference:p.error('Gradient gate requires completed identity engineering reference')
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min engineering budget')));signal.alarm(1800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<12000:raise RuntimeError('At least12GB free required')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.check_int_temporal_feature']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

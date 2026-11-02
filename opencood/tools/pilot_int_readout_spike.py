"""Matched two-convolution readout adaptation, fixed scan and update budget."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.models.sub_modules.int_temporal_feature import detach_temporal_state
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.pilot_int_spatial_spike import support_histogram
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils

TARGETS=ReadoutSpikeAnchor.trainable_names


def frozen_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name not in TARGETS:
            result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@contextmanager
def readout_capture(model):
    result={};layers=model.core.readout.layers[0].double_conv
    def common(module,inputs):result['common']=inputs[0]
    def target(module,inputs,output):result['output']=output
    hooks=[layers[0].register_forward_pre_hook(common),layers[3].register_forward_hook(target)]
    try:
        yield result
        assert set(result)=={'common','output'}
    finally:
        for hook in hooks:hook.remove()


def same_ann_cache(a,b):
    for k in ('value','coarse'):assert torch.equal(a[k],b[k])
    assert torch.equal(a['anchor']['value'],b['anchor']['value'])
    assert a['anchor']['timestamp_us']==b['anchor']['timestamp_us'] and a['anchor']['skipped']==b['anchor']['skipped']


def supervised_updates(rows):
    ends=window_ends(rows,4);labelled=False;result=[]
    for i,row in enumerate(rows):
        labelled=labelled or row['supervised']
        if i in ends:
            if labelled:result.append(row['frame'])
            labelled=False
    return result


def scaled_backward(model,loss,scale=4096.):
    """Fixed loss scale for FP16 recurrent storage; never silently skip an update."""
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite detection/local loss')
    (loss*scale).backward()
    parameters=dict(model.named_parameters())
    assert all(p.grad is None for n,p in parameters.items() if n not in TARGETS)
    opened=[parameters[n] for n in TARGETS]
    for p in opened:
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise RuntimeError('Nonfinite/missing scaled gradient')
        p.grad.div_(scale)
    norm=torch.nn.utils.clip_grad_norm_(opened,10.)
    if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite unscaled gradient norm')
    return dict(gradient_norm_before_clip=float(norm),
        gradient_l1_after_clip={n:float(parameters[n].grad.abs().sum()) for n in TARGETS})


def evaluation_arms(strict,models):
    # Every arm has independent recurrent state. Reset may reuse weights only.
    arms=dict(strict=strict,**models)
    for mode in ('lif','leaky'):
        arms[mode+'_reset']=models[mode]
        control=copy.deepcopy(models[mode]);control.activation_memory.storage_dtype=torch.float32
        assert model_digest(control)==model_digest(models[mode])
        arms[mode+'_fp32']=control
    return arms


def evaluate(models,base,cfg,rows,out,tag,identity=False):
    states={n:None for n in models};base_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    hashes={n:model_digest(m) for n,m in models.items()}
    dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={};outputs={}
            for name,model in models.items():
                if name!='strict':model.reset_activation_memory=name.endswith('_reset')
                with readout_capture(model) as observed:
                    (pred,states[name],info),cost=measured_step(model,x,states[name],meta)
                preds[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                entry=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=info.get('readout_activation'))
                if name=='strict':common=observed['common']
                else:
                    assert cost==record['arms']['strict']['core_conv_flops'] and info['late_refresh']==record['arms']['strict']['refresh']
                    assert torch.equal(observed['common'],common);same_ann_cache(states[name],states['strict'])
                    entry.update(input_support_4_6_9=support_histogram(model.last_input),output_support_4_6_9=support_histogram(model.last_output),
                        target_input_mse=float((model.last_input-model.last_output).square().mean()))
                    outputs[name]=model.last_output
                    if identity and name=='ann':assert all(torch.equal(v,preds['strict'][k]) for k,v in pred.items())
                    if name.endswith('_fp32'):
                        source=name[:-5];delta=model.last_output-outputs[source]
                        entry['fp32_vs_fp16']=dict(output_support_disagreements=int(torch.count_nonzero((model.last_output!=0)!=(outputs[source]!=0))),
                            elements=model.last_output.numel(),output_mse=float(delta.square().mean()),
                            output_max_error=float(delta.abs().max()),prediction_max_error=max(float((v-preds[source][k]).abs().max()) for k,v in pred.items()))
                expected=22579200+(0 if name in ('strict','ann') else (25804800 if name.endswith('_fp32') else 12902400))
                assert info['cache_bytes']==expected
                record['arms'][name]=entry
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag+'_evaluation',scans=i+1,total=len(rows)))
    assert len(records)==len(rows)==256
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/(tag+'_ap_records.json'),ap)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),adaptation_parameters=0 if n=='strict' else 1475072,
        core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/len(records),cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0],model_step_ms_mean=None,
        timing_invalid_reason='Instrumented shared-GPU multi-arm evaluation; no valid timing') for n,m in models.items()}
    for name,model in models.items():
        if name!='strict':
            model.reset_activation_memory=False;model.last_input=model.last_output=model.last_common_input=None
    return dict(scans=len(records),labels=sum(r['supervised'] for r in records),metrics=metrics),ap


def train_one(model,teacher,base,cfg,rows,out,name,criterion,contract,expected_updates,preflight=False):
    model.eval();model.reset_activation_memory=False;before=frozen_digest(model);teacher_before=model_digest(teacher)
    assert set(n for n,p in model.named_parameters() if p.requires_grad)==set(TARGETS)
    parameters=dict(model.named_parameters());initial={n:parameters[n].detach().clone() for n in TARGETS}
    assert sum(p.numel() for n,p in parameters.items() if n in TARGETS)==1475072
    opened=[parameters[n] for n in TARGETS]
    optimizer=torch.optim.AdamW(opened,lr=.0003,weight_decay=.0001);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=teacher_state=base_state=None;det=[];local=[];updates=[];labels=0;windows=[]
    _,stream=loader(rows,cfg,2,303);tag=name+('_preflight' if preflight else '_train')
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];assert meta['frame']==rows[i]['frame'];ego=to_device(sample['ego'],torch.device('cuda'))
            with torch.no_grad():
                x,base_state=capture(base,ego,base_state,meta)
                with readout_capture(teacher) as tc:_,teacher_state,teacher_info=teacher.step_spatial(x,teacher_state,meta)
            with readout_capture(model) as sc:pred,state,info=model.step_spatial(x,state,meta,detach_state=False)
            assert torch.equal(sc['common'],tc['common']);same_ann_cache(state,teacher_state)
            assert info['late_refresh']==teacher_info['late_refresh']
            kd=(sc['output']-tc['output'].detach()).square().mean()/(tc['output'].detach().square().mean()+1e-6)
            assert bool(torch.isfinite(kd));local.append(kd)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=info['late_refresh'],memory=info['readout_activation'],
                local_loss=float(kd.detach()),common_readout_input_exact=True,optimizer_step=False,loss_scale=4096.)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));det.append(loss);labels+=1;record['detection_loss']=float(loss.detach())
            if i in ends:
                if det:
                    detection=torch.stack(det).mean();distill=torch.stack(local).mean();loss=detection+distill
                    record.update(scaled_backward(model,loss));record.update(window_detection_loss=float(detection.detach()),window_local_loss=float(distill.detach()),window_total_loss=float(loss.detach()))
                    if preflight:
                        log.write(json.dumps(record)+'\n')
                        assert meta['frame']==expected_updates[0] and all(torch.equal(parameters[n],initial[n]) for n in TARGETS)
                        assert frozen_digest(model)==before and model_digest(teacher)==teacher_before
                        model.zero_grad(set_to_none=True);model.last_input=model.last_output=model.last_common_input=None
                        return dict(status='PASSED',scans=i+1,labels=labels,frame=meta['frame'],optimizer_steps=0,weights_bn_scale_unchanged=True,gradient=record)
                    optimizer.step();record['optimizer_step']=True;updates.append(meta['frame']);windows.append(record['window_total_loss'])
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);det=[];local=[];state=detach_temporal_state(state)
            log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=512,updates=len(updates)))
    assert not preflight and i+1==512 and labels==232 and updates==expected_updates and len(updates)==71 and not det and not local
    assert frozen_digest(model)==before and model_digest(teacher)==teacher_before
    assert all(not torch.equal(initial[n],parameters[n]) for n in TARGETS)
    path=out/(name+'_readout.pth')
    torch.save(dict(targets={n:parameters[n].detach().cpu() for n in TARGETS},scale=model.activation_memory.scale.detach().cpu(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=71,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Completed fixed pass; any continuation separately registered and starts from empty stream state'),path)
    model.last_input=model.last_output=model.last_common_input=None
    return dict(scans=512,labels=labels,optimizer_steps=71,frozen_weights_bn_scale_unchanged=True,teacher_unchanged=True,common_input_and_ann_cache_exact=True,
        target_change_l2={n:float((parameters[n].detach()-initial[n]).norm()) for n in TARGETS},mean_window_loss=sum(windows)/len(windows),checkpoint_sha256=digest(path))


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    gate=Path(a.gate);g=json.loads((gate/'report.json').read_text());cal=json.loads((gate/'calibration.json').read_text())
    verified=json.loads(Path(a.gate_verification).read_text())
    assert verified['status']=='VERIFIED' and verified['source_sha256']['report.json']==digest(gate/'report.json')
    assert g['status']=='COMPLETED' and g['gradient_gate_passed'] and g['common_input_exact']
    assert g['checkpoint_sha256']==cal['checkpoint_sha256']==digest(Path(a.checkpoint))
    reference=Path(a.reference);prior=json.loads((reference/'report.json').read_text());prior_contract=json.loads((reference/'contract.json').read_text())
    assert prior['status']=='COMPLETED' and prior['checkpoint_sha256']==g['checkpoint_sha256']
    data=json.loads(Path(a.data_audit).read_text());assert data['status']=='VERIFIED' and data['frames']==768
    for row in data['files']:assert digest(Path(row['relocated_path']))==row['sha256'],row['relocated_path']
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    training,evaluation=rows[:512],rows[512:768]
    assert sorted({r[k] for r in rows[:768] for k in ('pcd','label') if r[k]})==sorted(r['relocated_path'] for r in data['files'])
    assert [r['frame'] for r in training]==prior_contract['train_frames'] and [r['frame'] for r in evaluation]==prior_contract['evaluation_frames']
    expected=supervised_updates(training)
    assert expected==[json.loads(line)['frame'] for line in (reference/'lif_train_frames.jsonl').read_text().splitlines() if json.loads(line)['optimizer_step']]
    assert len(expected)==71 and cal['frames']==[r['frame'] for r in training[:256]]
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);strict=AnchoredLatePyramidDetector(base)
    models={n:ReadoutSpikeAnchor(base,n,cal['scale'],torch.float16).cuda().eval() for n in ('ann','lif','leaky')}
    assert all(torch.equal(v,models['ann'].state_dict()[k]) for model in models.values() for k,v in model.state_dict().items())
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_report_sha256=digest(gate/'report.json'),calibration_sha256=digest(gate/'calibration.json'),gate_verification_sha256=digest(Path(a.gate_verification)),
        data_audit_sha256=digest(Path(a.data_audit)),reference_report_sha256=digest(reference/'report.json'),train_frames=prior_contract['train_frames'],evaluation_frames=prior_contract['evaluation_frames'],expected_update_frames=expected,
        split='train',seed=303,epochs=1,tbptt=4,optimizer='AdamW',lr=.0003,weight_decay=.0001,clip_norm=10.,local_loss_weight=1.,local_loss_epsilon=1e-6,loss_scale=4096.,
        loss='Labelled-frame detection mean plus all-window-frame mean readout last-conv pre-BN normalized MSE; independent detached strict teacher. Context-only windows do not update.',
        parameters=32756052,adaptation_parameters=1475072,targets=TARGETS,tau=.2,threshold=1.,scale=cal['scale'],storage_dtype='float16',compute_dtype='float32',
        schedule='Sequential ann, lif, leaky on one GPU; 512 scans each; identical71 supervised update frames; no search/skip/continuation',
        evaluation='Disjoint train scan prefixes, possibly correlated scenes; empty own state. Reset only auxiliary membrane. FP32 controls share respective learned weights, no separate training.',timing_valid=False)
    write_json(out/'contract.json',contract)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    write_json(out/'status.json',dict(status='RUNNING',phase='detection_gradient_preflight'))
    preflight={n:train_one(m,strict,base,cfg,training,out,n,criterion,contract,expected,True) for n,m in models.items()}
    write_json(out/'preflight.json',preflight)
    initial,initial_ap=evaluate(evaluation_arms(strict,models),base,cfg,evaluation,out,'initial',True)
    assert initial_ap['strict']==initial_ap['ann']
    assert json.loads((out/'initial_ap_records.json').read_text())['strict']==json.loads((reference/'final_ap_records.json').read_text())['strict']
    learned={n:train_one(m,strict,base,cfg,training,out,n,criterion,contract,expected) for n,m in models.items()}
    final,final_ap=evaluate(evaluation_arms(strict,models),base,cfg,evaluation,out,'final');assert final_ap['strict']==initial_ap['strict']
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],initial=initial,preflight=preflight,training=learned,final=final,
        initial_ann_exact=True,strict_raw_ap_replayed=True,trained=True,timing_valid=False,torch_version=torch.__version__,process_cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),
        boundary='One-pass two-original-convolution adaptation, fixed dynamics/scale; learned upstream event encoding. Train-prefix evaluation, not full val or independent test. Dense GPU Conv, no valid timing/physical energy measurement.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','gate','gate-verification','reference','data-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min matched readout pilot budget')));signal.alarm(5400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free, process cap50%; no valid timing on shared GPU')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.pilot_int_readout_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

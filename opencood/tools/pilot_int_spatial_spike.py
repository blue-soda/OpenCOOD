"""Matched ANN/LIF/leaky single-convolution adaptation, fixed one-pass budget."""
import argparse
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
from opencood.models.sub_modules.int_spatial_spike import SpatialSpikeAnchor,deliveries3x3
from opencood.models.sub_modules.int_temporal_feature import detach_temporal_state
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils

TARGET='core.coarse_encoder.3.conv2.weight'


def frozen_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name!=TARGET:result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def target_step(model,x,state,meta,detach_state):
    captured=[];hook=model.core.coarse_encoder[3].conv2.register_forward_hook(lambda m,inputs,output:captured.append((inputs[0],output)))
    try:result=model.step_spatial(x,state,meta,detach_state=detach_state)
    finally:hook.remove()
    assert len(captured)==1
    return result,captured[0]


def support_histogram(value):
    active=value!=0;n=int(active.sum());corners=int(active[...,0,0].sum()+active[...,0,-1].sum()+active[...,-1,0].sum()+active[...,-1,-1].sum())
    edges=int(active[...,0,1:-1].sum()+active[...,-1,1:-1].sum()+active[...,1:-1,0].sum()+active[...,1:-1,-1].sum())
    result=[corners,edges,n-corners-edges]
    assert sum(k*v for k,v in zip((4,6,9),result))*256==deliveries3x3(active,256)
    return result


def evaluate(models,base,cfg,rows,out,tag,identity=False):
    states={n:None for n in models};base_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for name,model in models.items():
                if name!='strict':model.reset_activation_memory=name.endswith('_reset')
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);preds[name]=pred
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                entry=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=info.get('spatial_activation'))
                if name!='strict':
                    assert cost==record['arms']['strict']['core_conv_flops'] and info['late_refresh']==record['arms']['strict']['refresh']
                    assert torch.equal(model.last_input,models['ann'].last_input)
                    entry.update(input_support_4_6_9=support_histogram(model.last_input),output_support_4_6_9=support_histogram(model.last_output),
                        target_input_mse=float((model.last_input-model.last_output).square().mean()))
                    if identity and name=='ann':assert all(torch.equal(v,preds['strict'][k]) for k,v in pred.items())
                record['arms'][name]=entry
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(record);log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag+'_evaluation',scans=i+1,total=256))
    assert len(records)==256
    write_json(out/(tag+'_ap_records.json'),ap)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),adaptation_parameters=0 if n=='strict' else 589824,
        core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],scale_buffer_bytes=0 if n=='strict' else 1024,
        ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0],model_step_ms_mean=None,
        timing_invalid_reason='Instrumented multi-arm train-subset evaluation; no valid timing') for n,m in models.items()}
    for name,model in models.items():
        if name!='strict':model.reset_activation_memory=False
    return dict(scans=256,labels=sum(r['supervised'] for r in records),metrics=metrics),ap


def train_one(model,teacher,base,cfg,rows,out,name,criterion,contract,expected_updates):
    model.eval();model.reset_activation_memory=False;before=frozen_digest(model);teacher_before=model_digest(teacher)
    assert [n for n,p in model.named_parameters() if p.requires_grad]==[TARGET]
    weight=model.core.coarse_encoder[3].conv2.weight;initial=weight.detach().clone();assert weight.numel()==589824
    optimizer=torch.optim.AdamW([weight],lr=.0003,weight_decay=.0001);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=teacher_state=base_state=None;det=[];local=[];updates=[];labels=0;windows=[]
    _,stream=loader(rows,cfg,2,303)
    with (out/(name+'_train_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];assert meta['frame']==rows[i]['frame'];ego=to_device(sample['ego'],torch.device('cuda'))
            with torch.no_grad():
                x,base_state=capture(base,ego,base_state,meta)
                (_,teacher_state,_),(teacher_input,teacher_output)=target_step(teacher,x,teacher_state,meta,True)
            (pred,state,info),(_,student_output)=target_step(model,x,state,meta,False)
            assert torch.equal(model.last_input,teacher_input)
            kd=(student_output-teacher_output.detach()).square().mean()/(teacher_output.detach().square().mean()+1e-6)
            assert bool(torch.isfinite(kd));local.append(kd)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=info['late_refresh'],memory=info['spatial_activation'],
                local_loss=float(kd.detach()),teacher_input_exact=True,optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));det.append(loss);labels+=1;record['detection_loss']=float(loss.detach())
            if i in ends:
                if det:
                    detection=torch.stack(det).mean();distill=torch.stack(local).mean();loss=detection+distill;loss.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if n!=TARGET)
                    norm=torch.nn.utils.clip_grad_norm_([weight],10.);assert bool(torch.isfinite(norm)) and bool(torch.isfinite(weight.grad).all())
                    record.update(optimizer_step=True,window_detection_loss=float(detection.detach()),window_local_loss=float(distill.detach()),
                        window_total_loss=float(loss.detach()),gradient_norm_before_clip=float(norm),gradient_l1_after_clip=float(weight.grad.abs().sum()))
                    optimizer.step();updates.append(meta['frame']);windows.append(record['window_total_loss'])
                optimizer.zero_grad(set_to_none=True);det=[];local=[];state=detach_temporal_state(state)
            log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='train_'+name,scans=i+1,total=512,updates=len(updates)))
    assert i+1==512 and labels==232 and updates==expected_updates and len(updates)==71 and not det and not local
    assert frozen_digest(model)==before and model_digest(teacher)==teacher_before and bool(torch.isfinite(weight).all())
    assert not torch.equal(initial,weight)
    path=out/(name+'_target.pth')
    torch.save(dict(target_weight=weight.detach().cpu(),scale=model.activation_memory.scale.detach().cpu(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=71,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Completed fixed pass; any continuation separately registered and starts from empty stream state'),path)
    return dict(scans=512,labels=labels,optimizer_steps=71,frozen_weights_bn_scale_unchanged=True,teacher_unchanged=True,common_input_exact=True,
        target_change_l2=float((weight.detach()-initial).norm()),mean_window_loss=sum(windows)/len(windows),checkpoint_sha256=digest(path))


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    gate=Path(a.gate);g=json.loads((gate/'report.json').read_text());cal=json.loads((gate/'calibration.json').read_text())
    reference=Path(a.reference);prior=json.loads((reference/'report.json').read_text());prior_contract=json.loads((reference/'contract.json').read_text())
    assert g['status']=='COMPLETED' and g['gradient_gate_passed'] and g['common_input_exact']
    assert prior['status']=='COMPLETED' and prior['matched_reference_verified']
    assert g['checkpoint_sha256']==prior['checkpoint_sha256']==cal['checkpoint_sha256']==digest(Path(a.checkpoint))
    torch.cuda.set_per_process_memory_fraction(.65)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    training,evaluation=rows[:512],rows[512:768]
    assert [r['frame'] for r in training]==prior_contract['train_frames'] and [r['frame'] for r in evaluation]==prior_contract['evaluation_frames']
    expected=[json.loads(line)['frame'] for line in (reference/'lif_train_frames.jsonl').read_text().splitlines() if json.loads(line)['optimizer_step']]
    assert len(expected)==71 and cal['frames']==[r['frame'] for r in training[:256]]
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model'],strict=True);strict=AnchoredLatePyramidDetector(base)
    models={n:SpatialSpikeAnchor(base,n,cal['scale']).cuda().eval() for n in ('ann','lif','leaky')}
    assert all(torch.equal(v,models['ann'].state_dict()[k]) for model in models.values() for k,v in model.state_dict().items())
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_report_sha256=digest(gate/'report.json'),calibration_sha256=digest(gate/'calibration.json'),
        reference_report_sha256=digest(reference/'report.json'),train_frames=prior_contract['train_frames'],evaluation_frames=prior_contract['evaluation_frames'],expected_update_frames=expected,
        split='train',seed=303,epochs=1,tbptt=4,optimizer='AdamW',lr=.0003,weight_decay=.0001,clip_norm=10.,local_loss_weight=1.,local_loss_epsilon=1e-6,
        loss='Labelled-frame detection mean plus all-window-frame mean pre-BN target-conv normalized MSE; teacher detached. Context-only windows do not update.',
        parameters=32756052,adaptation_parameters=589824,target=TARGET,tau=.2,threshold=1.,scale=cal['scale'],
        schedule='Sequential ann, lif, leaky on one GPU; 512 scans each; identical71 supervised update frames; no search or continuation',
        evaluation='Disjoint train scan prefixes, possibly correlated scenes; each arm starts empty. Reset is fixed-weight auxiliary-membrane intervention; ANN recurrent state remains.',timing_valid=False)
    write_json(out/'contract.json',contract)
    arms=dict(strict=strict,**models);arms.update(lif_reset=models['lif'],leaky_reset=models['leaky'])
    initial,initial_ap=evaluate(arms,base,cfg,evaluation,out,'initial',True)
    assert initial_ap['strict']==initial_ap['ann']
    assert json.loads((out/'initial_ap_records.json').read_text())['strict']==json.loads((reference/'final_ap_records.json').read_text())['strict']
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();learned={}
    for name,model in models.items():learned[name]=train_one(model,strict,base,cfg,training,out,name,criterion,contract,expected)
    final,final_ap=evaluate(arms,base,cfg,evaluation,out,'final');assert final_ap['strict']==initial_ap['strict']
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],initial=initial,training=learned,final=final,
        initial_ann_exact=True,strict_raw_ap_replayed=True,trained=True,timing_valid=False,
        boundary='Single-pass adaptation of one existing convolution under fixed encoding; not end-to-end membrane training, full validation or independent test. Actual Conv dense; ideal operations are not total energy.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','gate','reference','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min matched pilot budget')));signal.alarm(5400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<20000:raise RuntimeError('Requires20GB free on one idle GPU')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.pilot_int_spatial_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

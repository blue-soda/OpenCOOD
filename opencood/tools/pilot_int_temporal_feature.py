"""Single-GPU sequential matched adapter pilot; disjoint train scan prefixes."""
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
from torch import nn
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_temporal_feature import DifferentiableTemporalFeatureAnchor,detach_temporal_state
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def frozen_digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('core.temporal_feature.'):
            result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def evaluate(models,base,cfg,rows,out,tag,identity=False):
    states={n:None for n in models};base_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for sample in stream:
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for name,model in models.items():
                if name!='strict':model.reset_feature_memory=name.endswith('_reset')
                (pred,states[name],info),cost=measured_step(model,x,states[name],meta);preds[name]=pred
                assert all(torch.isfinite(v).all() for v in pred.values())
                if name!='strict':
                    assert info['late_refresh']==record['arms']['strict']['refresh']
                    assert cost==record['arms']['strict']['core_conv_flops']+1032192000
                    if identity:assert all(torch.equal(v,preds['strict'][k]) for k,v in pred.items())
                record['arms'][name]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['late_refresh'],memory=info.get('feature_memory'))
            if meta['supervised']:
                data=dict(ego=ego);gt=dataset.post.generate_gt_bbx(data)
                for name,pred in preds.items():
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(record);log.write(json.dumps(record)+'\n')
    assert len(records)==256
    write_json(out/(tag+'_ap_records.json'),ap)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),adaptation_parameters=0 if n=='strict' else 82016,
        core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],
        ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0],model_step_ms_mean=None,
        timing_invalid_reason='Multi-arm instrumented train-subset evaluation, no valid timing') for n,m in models.items()}
    for name,model in models.items():
        if name!='strict':model.reset_feature_memory=False
    return dict(scans=256,labels=sum(r['supervised'] for r in records),metrics=metrics),ap


def train_one(model,base,cfg,rows,out,name,criterion,contract):
    model.eval();model.reset_feature_memory=False
    before=frozen_digest(model);memory=model.core.temporal_feature;memory.record_channels=True
    params=list(memory.parameters());assert sum(p.numel() for p in params)==82016
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.0001)
    ends=window_ends(rows,4);state=base_state=None;losses=[];updates=labels=0;windows=[]
    optimizer.zero_grad(set_to_none=True);_,stream=loader(rows,cfg,2,303)
    with (out/(name+'_train_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];assert meta['frame']==rows[i]['frame'];ego=to_device(sample['ego'],torch.device('cuda'))
            with torch.no_grad():x,base_state=capture(base,ego,base_state,meta)
            pred,state,info=model.step_spatial(x,state,meta,detach_state=False)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=info['late_refresh'],memory=info['feature_memory'],optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert torch.isfinite(loss)
                losses.append(loss);labels+=1;record['supervised_loss']=float(loss.detach())
            if i in ends:
                if losses:
                    loss=torch.stack(losses).mean();loss.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('core.temporal_feature.'))
                    norm=torch.nn.utils.clip_grad_norm_(params,10.);assert torch.isfinite(norm)
                    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                    record.update(optimizer_step=True,window_loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                        gradients_l1={n:0. if p.grad is None else float(p.grad.abs().sum()) for n,p in memory.named_parameters()})
                    optimizer.step();updates+=1;windows.append(record['window_loss'])
                optimizer.zero_grad(set_to_none=True);losses=[];state=detach_temporal_state(state)
            log.write(json.dumps(record)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='train_'+name,scans=i+1,total=512,updates=updates))
    assert i+1==512 and labels==sum(r['supervised'] for r in rows) and not losses and updates>0
    assert frozen_digest(model)==before
    assert all(bool(torch.isfinite(p).all()) for p in params)
    assert bool((memory.readout.weight!=0).any())
    memory.record_channels=False
    checkpoint=out/(name+'_adapter.pth')
    torch.save(dict(adapter=memory.state_dict(),optimizer=optimizer.state_dict(),contract=contract,mode=name,updates=updates,
        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Completed fixed one-pass pilot; any continuation starts a newly registered pass from empty stream state'),checkpoint)
    return dict(scans=512,labels=labels,optimizer_steps=updates,supervised_windows=len(windows),mean_window_loss=sum(windows)/len(windows),
        frozen_parameters_and_bn_unchanged=True,readout_weight_l1=float(memory.readout.weight.detach().abs().sum()),checkpoint_sha256=digest(checkpoint))


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    gate=Path(a.gradient_gate);prior=json.loads(gate.read_text())
    assert prior['status']=='COMPLETED' and prior['gradient_gate_passed'] and prior['identity_records_equal_R102']
    assert prior['checkpoint_sha256']==digest(Path(a.checkpoint))
    torch.cuda.set_per_process_memory_fraction(.6)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    assert len(rows)>=768
    training,evaluation=rows[:512],rows[512:768]
    assert not {r['frame'] for r in training}&{r['frame'] for r in evaluation}
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model'],strict=True)
    models={n:DifferentiableTemporalFeatureAnchor(base,n).cuda().eval() for n in ('lif','leaky')}
    assert all(torch.equal(v,models['leaky'].core.temporal_feature.state_dict()[k]) for k,v in models['lif'].core.temporal_feature.state_dict().items())
    strict=AnchoredLatePyramidDetector(base)
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gradient_gate_sha256=digest(gate),split='train',seed=303,
        train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],
        train_scans=512,evaluation_scans=256,epochs=1,optimizer='AdamW',lr=.0003,weight_decay=.0001,clip_norm=10.,tbptt=4,
        loss='Mean PointPillarDirLoss over labelled frames per window; no distillation; context-only windows detach without update',
        adaptation_parameters=82016,parameters=32838068,tau_u=.2,tau_r=.1,threshold=1.,input_gain=.25,
        schedule='Sequential lif then leaky on same GPU, identical frames/windows/initialization, frozen backbone and BN, no LR/AP search',
        evaluation='Disjoint train scan prefixes, possibly correlated scenes; each arm starts with empty stream state, no train-state carryover. Reset is fixed-weight intervention, not separately trained.',
        precision='FP32, TF32 disabled',timing_valid=False)
    write_json(out/'contract.json',contract)
    write_json(out/'status.json',dict(status='RUNNING',phase='initial_evaluation'))
    initial,initial_ap=evaluate(dict(strict=strict,**models),base,cfg,evaluation,out,'initial',True)
    assert initial_ap['strict']==initial_ap['lif']==initial_ap['leaky']
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();learned={}
    for name,model in models.items():learned[name]=train_one(model,base,cfg,training,out,name,criterion,contract)
    assert learned['lif']['optimizer_steps']==learned['leaky']['optimizer_steps']
    write_json(out/'status.json',dict(status='RUNNING',phase='final_evaluation'))
    final,final_ap=evaluate(dict(strict=strict,lif=models['lif'],leaky=models['leaky'],lif_reset=models['lif'],leaky_reset=models['leaky']),base,cfg,evaluation,out,'final')
    assert initial_ap['strict']==final_ap['strict']
    write_json(out/'report.json',dict(status='COMPLETED',split='train',train_scans=512,evaluation_scans=256,checkpoint_sha256=contract['checkpoint_sha256'],
        initial=initial,training=learned,final=final,initial_predictions_exact=True,strict_raw_ap_replayed=True,
        trained=True,timing_valid=False,boundary='Single-seed fixed-budget mechanism pilot on train subsets, not full val or independent test. Dense Conv plus continuous state; no energy/speed advantage claimed.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','gradient-gate','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min sequential pilot budget')));signal.alarm(5400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<20000:raise RuntimeError('Pilot requires20GB free; use one otherwise idle GPU')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.pilot_int_temporal_feature']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

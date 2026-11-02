"""Matched single-coarse-state learning pilot; one GPU, fixed disjoint prefixes."""
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
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.utils import eval_utils


PREFIX='core.coarse_update.'


def frozen_digest(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if not n.startswith(PREFIX):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def detach_state(state,cpu=False):
    return {k:(v.detach().cpu() if cpu else v.detach()) if torch.is_tensor(v) else v for k,v in state.items()}


def expected_updates(rows):
    ends=window_ends(rows,4);has_label=False;result=[]
    for i,r in enumerate(rows):
        has_label|=bool(r['supervised'])
        if i in ends:
            if has_label:result.append(r['frame'])
            has_label=False
    return result


def train_one(model,cfg,rows,out,name,criterion,contract,preflight=False):
    original_frozen=frozen_digest(model);before=model_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad==n.startswith(PREFIX) for n,p in model.named_parameters())
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=None;losses=[];updates=[];loss_values=[];labels=0
    _,stream=loader(rows,cfg,2,303);tag=name+('_preflight' if preflight else '_train')
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
            assert meta['frame']==rows[i]['frame']
            pred,state,info=model.step(ego,state,meta,detach_state=False)
            assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss))
                losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if losses:
                    objective=torch.stack(losses).mean();objective.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(PREFIX))
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in opened)
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()))
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==before
                        model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],scans=i+1,gradient_norm=float(norm),weights_unchanged=True)
                    optimizer.step();updates.append(meta['frame']);loss_values.append(float(objective.detach()));row['optimizer_step']=True
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%16==0:
                    path=out/('%s_update_%03d.pth'%(name,len(updates)))
                    torch.save(dict(coarse_update=model.core.coarse_update.state_dict(),optimizer=optimizer.state_dict(),
                        own_state=detach_state(state,cpu=True),next_scan=i+1,updates=updates,contract=contract,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),mode=name),path)
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows),updates=len(updates)))
    assert not preflight and len(updates)==71 and updates==expected_updates(rows) and labels==232
    assert frozen_digest(model)==original_frozen and model_digest(model)!=before
    path=out/(name+'_coarse_final.pth')
    torch.save(dict(coarse_update=model.core.coarse_update.state_dict(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
        resume='Fixed pass complete; any continuation separately registered, start from empty stream state'),path)
    return dict(scans=512,labels=labels,updates=len(updates),update_frames=updates,frozen_weights_bn_unchanged=True,
        checkpoint_sha256=digest(path),mean_window_loss=sum(loss_values)/len(loss_values))


def evaluate(models,base,cfg,rows,out,tag,identity=False):
    states={n:None for n in models};base_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    hashes={n:model_digest(m) for n,m in models.items()};dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta)
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta)
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                predictions[n]=pred;entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],reset=info['reset'])
            if identity:assert all(torch.equal(v,predictions['original'][k]) for k,v in predictions['frozen'].items())
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows)))
    assert len(records)==256 and sum(r['supervised'] for r in records)==192
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/(tag+'_ap_records.json'),ap)
    return dict(scans=256,labels=192,metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],
        model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}),ap


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=json.loads(Path(a.gate_verification).read_text());assert gate['status']=='VERIFIED' and gate['run']=='R-125'
    assert gate['checkpoint_sha256']==digest(Path(a.checkpoint))
    data=json.loads(Path(a.data_audit).read_text());assert data['status']=='VERIFIED' and data['frames']==768
    for r in data['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    torch.cuda.set_per_process_memory_fraction(.6)
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    training,evaluation=rows[:512],rows[512:768]
    assert len(expected_updates(training))==71 and sum(r['supervised'] for r in training)==232
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True)
    for p in base.parameters():p.requires_grad_(False)
    models={}
    for name in ('frozen','original','gate','leaky'):
        torch.manual_seed(303);model=TemporalPyramidDetector(base,'original' if name=='frozen' else name).cuda().eval()
        for n,p in model.named_parameters():p.requires_grad_(name!='frozen' and n.startswith(PREFIX))
        models[name]=model
    del teacher,saved
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),data_audit_sha256=digest(Path(a.data_audit)),
        gate_verification_sha256=digest(Path(a.gate_verification)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],
        expected_update_frames=expected_updates(training),seed=303,tbptt=4,epochs=1,lr=1e-4,weight_decay=1e-4,clip_norm=10.,
        optimizer='AdamW',loss='Mean labelled-frame detection loss within each window; no teacher loss or teacher history',
        trainable_prefix=PREFIX,gate_initialization='Copied candidate and z=.5 on valid history; not original-identity',
        leaky_tau=.2,dt='Actual adjacent timestamp difference; reset gap>.25',
        schedule='Sequential original/gate/leaky, same512 scans and71 update positions; frozen control no adaptation',
        boundary='Short learnability pilot, disjoint train scan prefixes but potentially correlated scenes; no convergence/full-val/independent-test or valid timing claim')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    preflight={n:train_one(models[n],cfg,training,out,n,criterion,contract,True) for n in ('original','gate','leaky')}
    write_json(out/'preflight.json',preflight)
    initial,initial_ap=evaluate(models,base,cfg,evaluation,out,'initial',True)
    learned={n:train_one(models[n],cfg,training,out,n,criterion,contract) for n in ('original','gate','leaky')}
    final,final_ap=evaluate(models,base,cfg,evaluation,out,'final')
    assert initial_ap['frozen']==final_ap['frozen'] and initial_ap['frozen']==initial_ap['original']
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,preflight=preflight,training=learned,final=final,
        checkpoint_sha256=contract['checkpoint_sha256'],frozen_raw_ap_exact=True,trained=True,timing_valid=False,torch_version=torch.__version__))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','data-audit','gate-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min matched gated-state pilot')));signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())
    if used>200:raise RuntimeError('Require idle GPU for memory-stable matched training')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.pilot_int_gated_state']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

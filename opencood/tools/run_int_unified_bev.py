"""Train a direct-readout recurrent BEV baseline, before adding selective updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def engineering(student,teacher,cfg,rows,out):
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:raise RuntimeError('INT engineering tests failed')
    calls=[0,0,0]
    def hit(index):
        def callback(module,inputs,output):calls[index]+=1
        return callback
    modules=[teacher.backbone.resnet,teacher.backbone.deblocks[0],teacher.shrink_conv]
    hooks=[m.register_forward_hook(hit(i)) for i,m in enumerate(modules)]
    student.eval();state=None;records=[]
    _,stream=loader(rows[:8],cfg,2,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'))
                pred,state,info=student.step(ego,state,sample['meta'])
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert pred['psm'].shape[-2:]==ego['anchor_box'].shape[:2]
                assert calls==[0,0,0]
                records.append(dict(frame=sample['meta']['frame'],reset=info['reset'],
                                    state_shape=list(state['value'].shape)))
    finally:
        for hook in hooks:hook.remove()
    assert not any('backbone' in name or 'feature_memory' in name for name,_ in student.named_modules())
    report=dict(status='PASSED',frames=records,original_spatial_calls=calls,
                state_bytes=info['state_bytes'],update_radius=student.core.update.radius,
                caveat='New model: finite predictions/shape/causality, not parity with the teacher. Crop equivalence is a unit test, not an active sparse policy.')
    write_json(out/'engineering.json',report)


def validate(a,student,teacher,cfg,rows,directory):
    a.stage='full'
    student.eval()
    torch.cuda.reset_peak_memory_stats()
    report=evaluate(a,teacher,cfg,rows,directory,student)
    report['scope']='Unified-state dense-update full validation'
    report['caveats']=[
        'All state cells are updated each frame. No selective-update or sparsity speedup is claimed.',
        'Own frozen Pillar encoder, observation/context, recurrent update, readout and heads only; no old GRU, ResNet or multiscale decoder.',
        'No teacher execution or teacher state in validation/inference. Model timing excludes IO/voxelization/postprocess.',
        'Concurrent training throughput is not a paired latency benchmark. Use independent same-GPU sequential evaluation after training.']
    return report


def train_epoch(a,student,teacher,cfg,rows,optimizer,criterion,epoch,directory,out):
    student.train();teacher.eval();state=teacher_state=None
    ends=window_ends(rows,4);distilled=[];supervised=[];losses=[];labels=updates=0
    _,stream=loader(rows,cfg,4,303+epoch)
    optimizer.zero_grad(set_to_none=True)
    with (directory/'train_frames.jsonl').open('x',buffering=1) as log:
        for index,(row,sample) in enumerate(zip(rows,stream)):
            meta=sample['meta'];assert row['frame']==meta['frame']
            ego=to_device(sample['ego'],torch.device('cuda'))
            # The teacher is an independent frozen stream and never seeds student state.
            with torch.no_grad():target,teacher_state,_=teacher.step(ego,teacher_state,meta,reset=meta['reset_before'])
            prediction,state,info=student.step(ego,state,meta,detach_state=False)
            kd=foreground_head_loss(prediction,target)
            if not bool(torch.isfinite(kd)):raise RuntimeError('Non-finite distillation')
            distilled.append(kd)
            record=dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],reset=info['reset'])
            if meta['supervised']:
                detection=criterion(prediction,ego['label_dict'])
                if not bool(torch.isfinite(detection)):raise RuntimeError('Non-finite supervised detection loss')
                supervised.append(detection);labels+=1
            if index in ends:
                loss=.2*torch.stack(distilled).mean()
                if supervised:loss=loss+torch.stack(supervised).mean()
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(student.core.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('Non-finite unified gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()));updates+=1
                record.update(window_end=True,loss=losses[-1],gradient_norm=float(norm))
                state=dict(state,value=state['value'].detach());distilled=[];supervised=[]
            log.write(json.dumps(record)+'\n')
            if (index+1)%200==0:
                status=dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,total_scans=len(rows),
                            labels=labels,updates=updates,mean_loss=float(np.mean(losses)) if losses else None)
                write_json(out/'status.json',status);print('TRAIN',json.dumps(status),flush=True)
    assert index+1==len(rows) and labels==sum(r['supervised'] for r in rows)
    assert updates==len(ends) and not distilled and not supervised
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.pillar_vfe.parameters())
    return dict(scans=len(rows),labels=labels,updates=updates,loss=float(np.mean(losses)))


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
        oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    teacher,cfg,val,provenance=setup(a)
    for p in teacher.parameters():p.requires_grad_(False)
    write_json(out/'provenance.json',provenance)
    runtime=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train_rows=json.loads(Path(runtime['manifest']).read_text())['train']
    torch.manual_seed(303);student=UnifiedBEVDetector(teacher,cfg).cuda()
    a.engineering=str(out/'engineering.json')
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location='cpu')
        assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
        assert saved['contract']['data_contract']==provenance['contract']
        student.load_state_dict(saved['model'],strict=True)
        engineering(student,teacher,cfg,val,out)
        if a.diagnostic:
            from opencood.tools.diagnose_int_unified_bev import diagnose
            report=diagnose(student,teacher,cfg,val,out)
        else:
            report=validate(a,student,teacher,cfg,val,out)
        report.update(checkpoint=a.checkpoint,checkpoint_sha256=digest(Path(a.checkpoint)))
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(status='COMPLETED'))
        return
    contract=dict(version='unified_bev_dense_v1',seed=303,epochs_min=5,epochs_max=a.epochs,
        data_contract=provenance['contract'],teacher_sha256=provenance['checkpoint']['sha256'],
        train_scans=len(train_rows),val_scans=len(val),tbptt=4,patience=5,min_delta=.001,
        optimizer='AdamW',initial_lr=.001,weight_decay=.0001,distillation_weight=.2,
        trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in student.parameters() if not p.requires_grad),
        architecture='Pillar64 frozen; observation96 stride2; coarse context32 at stride4; single128 state; radius2 gated update; pointwise256 readout and trainable detection heads',
        loss='Mean GT detection loss over labelled frames + .2 mean foreground head distillation over all frames in each TBPTT window; no teacher feature target',
        protocol='Dense every-frame autoregressive state writeback; student state only; no augmentation/delay; reset by segment or gap>.25s; no periodic teacher refresh',
        future_sparse='Radius2 crop parity only; no learned selector, skipped state cells or coarse-context caching in this baseline')
    optimizer=torch.optim.AdamW(student.core.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,
        threshold=.001,threshold_mode='abs',min_lr=1e-5)
    best=significant=-1.;stale=0;history=[];start=1
    if a.resume:
        source=Path(a.resume).parent
        assert json.loads((source/'status.json').read_text())['status']=='COMPLETED'
        assert json.loads((source/'report.json').read_text())['stop_reason']=='max_epochs_not_proof_of_convergence'
        saved=torch.load(a.resume,map_location='cpu')
        old=dict(saved['contract']);new=dict(contract);old.pop('epochs_max');new.pop('epochs_max')
        assert old==new and a.epochs>saved['epoch']
        assert saved['history']==json.loads((source/'history.json').read_text())
        student.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];significant=saved['significant'];stale=saved['stale'];start=saved['epoch']+1
        prior_best=json.loads((source/'best.json').read_text())
        assert prior_best['ap70']==best and digest(Path(prior_best['checkpoint']))==prior_best['sha256']
        write_json(out/'best.json',prior_best);write_json(out/'history.json',history)
        write_json(out/'resumption.json',dict(source=str(source),checkpoint_sha256=digest(Path(a.resume)),next_epoch=start))
    write_json(out/'training_contract.json',contract)
    engineering(student,teacher,cfg,val,out)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    for epoch in range(start,a.epochs+1):
        tick=time.monotonic();directory=out/('epoch_%03d'%epoch);directory.mkdir()
        result=train_epoch(a,student,teacher,cfg,train_rows,optimizer,criterion,epoch,directory,out)
        path=directory/'student.pth'
        torch.save(dict(model=student.state_dict(),epoch=epoch,contract=contract),str(path))
        write_json(out/'status.json',dict(status='RUNNING',phase='full_validation',epoch=epoch))
        report=validate(a,student,teacher,cfg,val,directory)
        report.update(epoch=epoch,checkpoint_sha256=digest(path));write_json(directory/'report.json',report)
        ap=report['metrics']['bev_ap70'];scheduler.step(ap)
        if ap>best:
            best=ap;write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(path),sha256=digest(path)))
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,ap70=ap,stale_epochs=stale,lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-tick,**result))
        write_json(out/'history.json',history)
        torch.save(dict(model=student.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            contract=contract,epoch=epoch,history=history,best=best,significant=significant,stale=stale,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),str(out/'latest.pth'))
        if epoch>=5 and stale>=5 and optimizer.param_groups[0]['lr']<=.0001:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),
        stop_reason='plateau' if stale>=5 and optimizer.param_groups[0]['lr']<=.0001 else 'max_epochs_not_proof_of_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--worker',action='store_true')
    p.add_argument('--diagnostic',action='store_true',help='256-frame teacher/student/reset diagnostic; requires checkpoint')
    a=p.parse_args();out=Path(a.output)
    if a.epochs<5 or (a.checkpoint and a.resume):p.error('At least five epochs; evaluation and resume are exclusive')
    if a.diagnostic and not a.checkpoint:p.error('Diagnostic requires an immutable checkpoint')
    if a.worker:
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('Requested GPU is busy')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.run_int_unified_bev']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=process.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid,output=str(out))))


if __name__=='__main__':main()

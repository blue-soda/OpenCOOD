"""Learn only a residual correction of the cached deep readout contribution."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout,SplitReadoutDetector
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.models.sub_modules.int_anchor_residual import refresh_reason,predict
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def repair_feature(repair,first,cache,meta,bounds):
    old,valid=warp_bev(cache['value'],cache['pose'],meta['pose'],bounds)
    return repair(torch.cat([first,old,valid.to(old.dtype)],1))


def capture_teacher(teacher):
    capture={}
    hooks=[teacher.backbone.resnet.layer0.register_forward_hook(lambda m,x,y:capture.update(first=y.detach())),
           teacher.shrink_conv.register_forward_pre_hook(lambda m,x:capture.update(decoded=x[0].detach()))]
    return capture,hooks


def modes_for(student,split,repair,cfg):
    return dict(full=SplitReadoutDetector(student,cfg,split,1),
        cached=SplitReadoutDetector(student,cfg,split,3),
        repaired=SplitReadoutDetector(student,cfg,split,3,repair))


def engineering(teacher,student,split,repair,cfg,rows):
    initial=copy.deepcopy(repair.state_dict())
    # Nonzero corrections are necessary to test training/deployment agreement.
    with torch.no_grad():repair.net[-1].weight.normal_(0,.001)
    modes=modes_for(student,split,repair,cfg);states={k:None for k in modes}
    costs={};source_state=anchor=None;capture,hooks=capture_teacher(teacher)
    calls=[0]*5
    def hit(i):
        def hook(m,x,y):calls[i]+=1
        return hook
    counted=[student.backbone.resnet.layer1,student.backbone.resnet.layer2,
             student.backbone.deblocks[1],student.backbone.deblocks[2],split.history]
    hooks += [m.register_forward_hook(hit(i)) for i,m in enumerate(counted)]
    partials=0;error=0.;depths=set()
    _,stream=loader(rows[:32],cfg,0,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                _,source_state,source_info=teacher.step(ego,source_state,meta)
                decoded=capture['decoded'];first=capture['first'];c=split.current_channels
                current=split.current(decoded[:,:c]);target=split.history(decoded[:,c:])
                reason=refresh_reason(anchor,meta,source_info['reset'],3)
                if reason:anchor=dict(value=target,pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
                value=target if reason else repair_feature(repair,first,anchor,meta,cfg['model']['args']['lidar_range'])
                expected=predict(student,split.fuse(current,value))
                if not reason:anchor=dict(anchor,frames=anchor['frames']+1)
                for name,mode in modes.items():
                    extras=torch.nn.ModuleList([split]+([repair] if name=='repaired' else []))
                    counter=ConvCounter(student,extras);before=list(calls);previous=states[name]
                    try:pred,states[name],info=mode.step(ego,states[name],meta)
                    finally:counter.close()
                    flag=info['local_partial'];key=name+('_partial' if flag else '_full')
                    if key in costs:assert costs[key]==counter.total
                    costs[key]=counter.total
                    if flag:
                        assert calls==before
                        assert states[name]['cache']['value'] is previous['cache']['value']
                    if name=='repaired':
                        error=max(error,max(float((expected[k]-pred[k]).abs().max()) for k in pred))
                        assert all(torch.allclose(expected[k],pred[k],atol=1e-4,rtol=1e-4) for k in pred)
                        if flag:partials+=1;depths.add(info['anchor_depth'])
        # Test gradient flow through frozen readout and head into repair.
        probe=repair_feature(repair,first,anchor,meta,cfg['model']['args']['lidar_range'])
        predict(student,split.fuse(current,probe))['rm'].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in repair.parameters())
        assert float(repair.net[-1].weight.grad.abs().sum())>0
        assert all(p.grad is None for m in (teacher,student,split) for p in m.parameters())
    finally:
        for hook in hooks:hook.remove()
        repair.load_state_dict(initial);repair.zero_grad(set_to_none=True)
    assert partials>0 and depths=={1,2}
    return dict(status='PASSED',frames=32,partial_frames=partials,train_deploy_max_error=error,
        partial_deep_decode_calls=0,gradient_checked=True,probe_weights_restored=True,conv_macs=costs)


def evaluate(student,split,repair,cfg,rows,out,costs,epoch):
    modes=modes_for(student,split,repair,cfg);states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    times={k:[] for k in modes};compute={k:0 for k in modes};partials={k:0 for k in modes}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            names=list(modes);offset=index%3;names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if index>=10:times[name].append(ms)
                flag=info['local_partial'];partials[name]+=flag
                compute[name]+=costs[name+('_partial' if flag else '_full')]
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200==0:write_json(out.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=index+1))
    assert index+1==len(rows) and partials['cached']==partials['repaired']
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(times[k])),p95_ms=float(np.percentile(times[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=partials[k]) for k in modes}
    assert abs(results['full']['ap70']-.7562138415770423)<1e-8
    assert abs(results['cached']['ap70']-.752781663200643)<1e-8
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        scope='Same GPU rotating synchronized model time, excluding IO/H2D/postprocess. Standard Conv2d/ConvTranspose2d MACs x2 only; excludes VFE Linear, warp, BN and activations. Dense small correction, no regional skipping.')
    write_json(out/'report.json',report);return report


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    teacher,cfg,val,provenance=setup(a)
    best,saved=completed_best(a.readout_run)
    assert best['sha256']=='8bbf94e975997b2803c4a4c443fee2258589276e427dbc0a2550de6b562d7a0d'
    assert saved['contract']['source_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    student=copy.deepcopy(teacher)
    student.shrink_conv=compact_shrink(teacher.shrink_conv,96)[0].cuda()
    student.shrink_conv.load_state_dict(saved['readout'],strict=True);student.eval()
    split=SplitCompactReadout(student.shrink_conv,student.backbone.deblocks[0][0].out_channels).cuda().eval()
    for m in (teacher,student,split):
        for p in m.parameters():p.requires_grad_(False)
    torch.manual_seed(303);repair=LightRepair(96,width=32).cuda().eval()
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train']
    contract=dict(readout=best,teacher=provenance['checkpoint'],data=provenance['contract'],seed=303,
        trainable_parameters=sum(p.numel() for p in repair.parameters()),width=32,channels=96,period=3,max_age=.25,
        train='Only partial frames; immutable original full deep-projection anchors. Frozen original encoder, compact split readout and head. No augmentation or artificial delays.',
        loss='GT on labelled frames + .2 foreground original INT head KD + .5 normalized feature KD of repaired deep projection against current full deep projection.',
        optimizer='AdamW lr1e-3 wd1e-4 clip5; AP plateau factor.5 patience1 threshold.001 min1e-6; min6 epochs, stale4 and lr<=1e-4 stop.',
        epochs_max=a.epochs,timeout_seconds=28800,inference='Whole-image radius1 LightRepair residual on deep contribution only; no recursive writeback; spatial readout remains dense.')
    write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',provenance)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(test.stdout);assert test.returncode==0
    gate=engineering(teacher,student,split,repair,cfg,train);write_json(out/'engineering.json',gate)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    optimizer=torch.optim.AdamW(repair.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-6)
    best_ap=significant=-1.;stale=0;history=[];bounds=cfg['model']['args']['lidar_range']
    for epoch in range(1,a.epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();source_state=anchor=None;losses=[];depths={1:0,2:0}
        capture,hooks=capture_teacher(teacher)
        _,stream=loader(train,cfg,4,303+epoch)
        try:
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():
                    target_pred,source_state,info=teacher.step(ego,source_state,meta)
                    first=capture['first'];decoded=capture['decoded'];c=split.current_channels
                    current=split.current(decoded[:,:c]);target=split.history(decoded[:,c:])
                    reason=refresh_reason(anchor,meta,info['reset'],3)
                    if reason:anchor=dict(value=target,pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
                if not reason:
                    optimizer.zero_grad(set_to_none=True)
                    corrected=repair_feature(repair,first,anchor,meta,bounds)
                    pred=predict(student,split.fuse(current,corrected))
                    loss=.2*foreground_head_loss(pred,target_pred)+.5*feature_distillation(corrected,target,target_pred)
                    if meta['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);loss.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(repair.parameters(),5.))
                    optimizer.step();losses.append(float(loss.detach()))
                    anchor=dict(anchor,frames=anchor['frames']+1);depths[anchor['frames']]+=1
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
        finally:
            for hook in hooks:hook.remove()
        assert index+1==len(train) and all(v>0 for v in depths.values())
        assert all(p.grad is None for m in (teacher,student,split) for p in m.parameters())
        path=directory/'repair.pth';torch.save(dict(repair=repair.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(student,split,repair,cfg,val,directory,gate['conv_macs'],epoch)
        ap=report['results']['repaired']['ap70'];scheduler.step(ap)
        if ap>best_ap:
            best_ap=ap;write_json(out/'best.json',dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path)))
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates_by_depth=depths,validation=report,lr=optimizer.param_groups[0]['lr'],stale=stale))
        write_json(out/'history.json',history)
        torch.save(dict(repair=repair.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,contract=contract),str(out/'latest.pth'))
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=1e-4:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),stop_reason='plateau' if epoch<a.epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','readout-run','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 1<=a.epochs<=12:p.error('Budget must be 1-12 epochs')
    if a.worker:
        def expired(s,f):raise TimeoutError('Training exceeded 8 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(28800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_split_repair']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=28800,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

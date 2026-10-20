"""Distill a compact spatial readout while preserving the mature current-E path."""
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
from opencood.models.sub_modules.int_compact_readout import compact_shrink,cached_features
from opencood.models.sub_modules.int_local_cache import LocalCacheDetector
from opencood.models.sub_modules.int_anchor_residual import refresh_reason,predict
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def wrappers(teacher,student,cfg):
    return dict(original=teacher,full=student,
        reference_cache=LocalCacheDetector(teacher,cfg,0.,period=3,repair_modules=cached_features(),repair_dense=True),
        compact_cache=LocalCacheDetector(student,cfg,0.,period=3,repair_modules=cached_features(),repair_dense=True))


def engineering(teacher,student,cfg,rows):
    modes=wrappers(teacher,student,cfg);states={k:None for k in modes};costs={};partials=0
    inputs={};calls=[0,0]
    hooks=[teacher.shrink_conv.register_forward_pre_hook(lambda m,x:inputs.update(reference=x[0])),
           student.shrink_conv.register_forward_pre_hook(lambda m,x:inputs.update(student=x[0]))]
    def hit(i):
        def callback(m,x,y):calls[i]+=1
        return callback
    hooks += [student.backbone.resnet.layer1.register_forward_hook(hit(0)),
              student.backbone.resnet.layer2.register_forward_hook(hit(1))]
    _,stream=loader(rows[:32],cfg,0,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                for name,mode in modes.items():
                    root=student if name in ('full','compact_cache') else teacher
                    counter=ConvCounter(root,torch.nn.ModuleList());before=list(calls)
                    try:prediction,states[name],info=mode.step(ego,states[name],meta)
                    finally:counter.close()
                    key=name+('_partial' if info.get('local_partial') else '_full')
                    if key in costs:assert costs[key]==counter.total
                    costs[key]=counter.total
                    if name=='compact_cache':
                        assert torch.equal(inputs['student'],inputs['reference'])
                        expected=predict(student,student.shrink_conv(inputs['reference']))
                        assert all(torch.equal(expected[k],prediction[k]) for k in prediction)
                        if info['local_partial']:
                            partials+=1;assert calls==before
        loss=student.shrink_conv(inputs['student'].detach()).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.shrink_conv.parameters())
        assert all(p.grad is None for n,p in student.named_parameters() if not n.startswith('shrink_conv.'))
        student.zero_grad(set_to_none=True)
    finally:
        for hook in hooks:hook.remove()
    assert partials>0
    return dict(status='PASSED',frames=32,same_cached_readout_input=True,gradient_checked=True,
                partial_frames_without_deep_calls=partials,conv_macs=costs)


def evaluate(teacher,student,cfg,rows,out,costs,epoch):
    modes=wrappers(teacher,student,cfg);states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    times={k:[] for k in modes};compute={k:0 for k in modes};partial={k:0 for k in modes}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            names=list(modes);names=names[index%4:]+names[:index%4]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if index>=10:times[name].append(ms)
                flag=bool(info.get('local_partial'));partial[name]+=flag
                compute[name]+=costs[name+('_partial' if flag else '_full')]
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200==0:write_json(out.parent/'status.json',dict(status='RUNNING',epoch=epoch,phase='validation',scans=index+1))
    assert index+1==len(rows)
    result={name:dict(ap70=eval_utils.calculate_ap(stats[name],.7)[0],
        ap50=eval_utils.calculate_ap(stats[name],.5)[0],mean_ms=float(np.mean(times[name])),
        p95_ms=float(np.percentile(times[name],95)),conv_gflops=2*compute[name]/len(rows)/1e9,
        partial_frames=partial[name]) for name in modes}
    assert abs(result['original']['ap70']-.758736104616861)<1e-8
    assert abs(result['reference_cache']['ap70']-.7532678498056796)<1e-8
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=result,
        scope='Rotating same-GPU model latency, excludes IO/H2D/postprocess. Standard Conv2d/ConvTranspose2d MACs x2 only, excluding VFE Linear/warps/BN/activations. No sparse region execution yet.')
    write_json(out/'report.json',report);return report


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    teacher,cfg,val,provenance=setup(a)
    assert provenance['checkpoint']['sha256']=='c7bbee9b4fdefc031a61b240d6c504665d66c0c788610bb36c97d34118f8a6b8'
    for p in teacher.parameters():p.requires_grad_(False)
    student=copy.deepcopy(teacher)
    compact,energy=compact_shrink(teacher.shrink_conv,a.rank);student.shrink_conv=compact.cuda()
    for p in student.shrink_conv.parameters():p.requires_grad_(True)
    student.eval()
    args=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(args['manifest']).read_text())['train']
    contract=dict(rank=a.rank,seed=303,epochs_max=a.epochs,refresh_period=3,max_age=.25,
        retained_weight_energy=energy,trainable_parameters=sum(p.numel() for p in compact.parameters()),
        source_sha256=provenance['checkpoint']['sha256'],data_contract=provenance['contract'],
        initialization='Input-channel low-rank SVD of both mature shrink 3x3 convs; preserve bias, nonlinearities and spatial radius2.',
        frozen='Original GRU, Pillar, all spatial stages, multiscale deblocks and detection heads.',
        train='Both full-current and three-frame cached deep-feature views; equal mean loss per view; immutable anchors, current E retained.',
        loss='GT on labelled frames + .2 foreground head KD + .5 normalized feature KD from mature readout of the SAME input view.',
        optimizer='AdamW lr1e-4 wd1e-4 clip5; cached AP plateau factor.5 patience1 min1e-6; min6 epochs then stale4 with lr<=1e-5 stop.',
        budgets='12 epochs maximum, 8 hours; budget end is not convergence',augmentation=False,artificial_delay=False,
        cache='Aligned original stage2/3 states; no learned correction, no recursive writeback. Identity slicing replaces diagnostic zero convolutions; reference AP must reproduce.')
    write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',provenance)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=engineering(teacher,student,cfg,train);write_json(out/'engineering.json',gate)
    optimizer=torch.optim.AdamW(student.shrink_conv.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    capture={};hooks=[]
    def save(index):
        def callback(m,x,y):capture[index]=y.detach()
        return callback
    for i in range(3):hooks.append(getattr(teacher.backbone.resnet,'layer%d'%i).register_forward_hook(save(i)))
    hooks.append(teacher.shrink_conv.register_forward_pre_hook(lambda m,x:capture.update(pre=x[0].detach())))
    history=[];best=significant=-1.;stale=0
    try:
        for epoch in range(1,a.epochs+1):
            state=cache=None;losses=[];tick=time.monotonic();directory=out/('epoch_%03d'%epoch);directory.mkdir()
            _,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():
                    _,state,info=teacher.step(ego,state,meta)
                    fresh=capture['pre'];stages=[capture[i] for i in range(3)]
                    views=[fresh]
                    reason=refresh_reason(cache,meta,info['reset'],3)
                    if reason:cache=dict(values=stages[1:],pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
                    else:
                        warped=[warp_bev(x,cache['pose'],meta['pose'],cfg['model']['args']['lidar_range'])[0] for x in cache['values']]
                        views.append(teacher.backbone.decode_multiscale_feature([stages[0]]+warped))
                        cache=dict(cache,frames=cache['frames']+1)
                optimizer.zero_grad(set_to_none=True);loss=0
                for inputs in views:
                    with torch.no_grad():target_features=teacher.shrink_conv(inputs);target=predict(teacher,target_features)
                    features=student.shrink_conv(inputs);pred=predict(student,features)
                    term=.2*foreground_head_loss(pred,target)+.5*feature_distillation(features,target_features,target)
                    if meta['supervised']:term=term+criterion(pred,ego['label_dict'])
                    loss=loss+term/len(views)
                assert torch.isfinite(loss);loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(student.shrink_conv.parameters(),5.))
                optimizer.step();losses.append(float(loss.detach()))
                if (index+1)%200==0:
                    write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
                    print('TRAIN',epoch,index+1,np.mean(losses),flush=True)
            assert index+1==len(train)
            assert all(p.grad is None for n,p in student.named_parameters() if not n.startswith('shrink_conv.'))
            path=directory/'readout.pth';torch.save(dict(readout=student.shrink_conv.state_dict(),contract=contract,epoch=epoch),str(path))
            for hook in hooks:hook.remove()
            hooks=[]
            report=evaluate(teacher,student,cfg,val,directory,gate['conv_macs'],epoch)
            for i in range(3):hooks.append(getattr(teacher.backbone.resnet,'layer%d'%i).register_forward_hook(save(i)))
            hooks.append(teacher.shrink_conv.register_forward_pre_hook(lambda m,x:capture.update(pre=x[0].detach())))
            ap=report['results']['compact_cache']['ap70'];scheduler.step(ap)
            if ap>best:
                best=ap;write_json(out/'best.json',dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path)))
            if ap>significant+.001:significant=ap;stale=0
            else:stale+=1
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=report,seconds=time.monotonic()-tick,lr=optimizer.param_groups[0]['lr']))
            write_json(out/'history.json',history)
            torch.save(dict(readout=student.shrink_conv.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,contract=contract,epoch=epoch),str(out/'latest.pth'))
            if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=1e-5:break
    finally:
        for hook in hooks:hook.remove()
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),stop_reason='plateau' if epoch<a.epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--rank',type=int,choices=(96,160),required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 1<=a.epochs<=12:p.error('Budget must be 1-12 epochs')
    if a.worker:
        def expired(s,f):raise TimeoutError('Compact readout exceeded 8 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(28800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_compact_readout']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=28800,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

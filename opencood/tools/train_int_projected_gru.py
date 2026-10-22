"""Recover a projected GRU using its own recurrent state and deep anchors.

Only three factorized gates (including BN affine, fixed running statistics)
learn. Both full and existing cached views supervise the same fresh evidence.
"""
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
from opencood.models.sub_modules.int_projected_gru import projected_gru
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


def capture_features(model):
    capture={}
    hooks=[model.backbone.resnet.layer0.register_forward_hook(lambda m,x,y:capture.update(first=y)),
        model.shrink_conv.register_forward_pre_hook(lambda m,x:capture.update(decoded=x[0]))]
    return capture,hooks


def cached_view(model,split,repair,capture,anchor,meta,info,bounds):
    first=capture['first'];decoded=capture['decoded'];c=split.current_channels
    current=split.current(decoded[:,:c]);reason=refresh_reason(anchor,meta,info['reset'],3)
    if reason:
        projected=split.history(decoded[:,c:])
        anchor=dict(value=projected.detach(),pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
    else:
        old,valid=warp_bev(anchor['value'],anchor['pose'],meta['pose'],bounds)
        projected=repair(torch.cat([first,old,valid.to(old.dtype)],1))
        anchor=dict(anchor,frames=anchor['frames']+1)
    return predict(model,split.fuse(current,projected)),anchor,reason is None


def make_modes(base,student,split,repair,cfg):
    return dict(base_full=SplitReadoutDetector(base,cfg,split,1),base_cached=SplitReadoutDetector(base,cfg,split,3,repair),
        full=SplitReadoutDetector(student,cfg,split,1),cached=SplitReadoutDetector(student,cfg,split,3,repair))


def engineering(base,student,split,repair,cfg,rows):
    captured,hooks=capture_features(student);state=anchor=None
    modes=make_modes(base,student,split,repair,cfg);states={k:None for k in modes};costs={};maximum=0.;depths=set()
    _,stream=loader(rows[:32],cfg,0,303);bounds=cfg['model']['args']['lidar_range']
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                expected_full,state,info=student.step(ego,state,meta)
                expected_cached,anchor,partial=cached_view(student,split,repair,captured,anchor,meta,info,bounds)
                if partial:depths.add(anchor['frames'])
                for name,mode in modes.items():
                    root=base if name.startswith('base') else student
                    extras=[split]+([repair] if name.endswith('cached') else [])
                    counter=ConvCounter(root,torch.nn.ModuleList(extras));previous=states[name]
                    try:pred,states[name],detail=mode.step(ego,states[name],meta)
                    finally:counter.close()
                    flag=detail['local_partial'];key=name+('_partial' if flag else '_full')
                    if key in costs:assert costs[key]==counter.total
                    costs[key]=counter.total
                    if flag:assert states[name]['cache']['value'] is previous['cache']['value']
                    if name in ('full','cached'):
                        target=expected_full if name=='full' else expected_cached
                        for k in pred:
                            maximum=max(maximum,float((pred[k]-target[k]).abs().max()))
                            assert torch.allclose(pred[k],target[k],atol=1e-4,rtol=1e-4)
        # A noninitial state exercises reset/update/candidate gradients.
        probe_meta=dict(meta,timestamp_us=int(meta['timestamp_us'])+100000)
        pred,_,_=student.step(ego,state,probe_meta)
        pred['rm'].square().mean().backward()
        for name in ('reset_gate_conv','update_gate_conv','candidate_conv'):
            gate=getattr(student.feature_memory.fusion,name)
            assert float(gate[0].weight.grad.abs().sum())>0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())
        assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
        assert all(p.grad is None for m in (base,split,repair) for p in m.parameters())
    finally:
        for hook in hooks:hook.remove()
        student.zero_grad(set_to_none=True)
    assert depths=={1,2}
    return dict(status='PASSED',frames=32,train_deploy_max_error=maximum,conv_macs=costs,
        own_recurrent_state=True,immutable_own_anchors=True,three_gate_gradients=True,frozen_rest=True)


def evaluate(base,student,split,repair,cfg,rows,directory,costs,epoch,expected_base=None,scope=None):
    modes=make_modes(base,student,split,repair,cfg);states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    timing={k:[] for k in modes};compute={k:0 for k in modes};partials={k:0 for k in modes}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter();pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                if index>=10:timing[name].append(ms)
                assert all(torch.isfinite(v).all() for v in pred.values())
                flag=info['local_partial'];partials[name]+=flag;compute[name]+=costs[name+('_partial' if flag else '_full')]
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200==0:write_json(directory.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=index+1))
    assert index+1==len(rows) and partials['cached']==partials['base_cached']
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(timing[k])),p95_ms=float(np.percentile(timing[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=partials[k]) for k in modes}
    expected_base=expected_base or (.7562138415770423,.7553967658970537)
    assert abs(results['base_full']['ap70']-expected_base[0])<1e-8
    assert abs(results['base_cached']['ap70']-expected_base[1])<1e-8
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        scope=scope or 'Dense projected GRU. Current states and anchors belong to each model; no teacher injections. Same GPU rotating model latency excluding IO/H2D/postprocess; standard convolution MACs x2 only.')
    write_json(directory/'report.json',report);return report


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    base,cfg,val,source=setup(a);best,readout=completed_best(a.readout_run)
    initial=Path(a.repair_run);assert json.loads((initial/'status.json').read_text())['status']=='COMPLETED'
    repair_best=json.loads((initial/'best.json').read_text());assert digest(repair_best['checkpoint'])==repair_best['sha256']
    assert repair_best['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    saved=torch.load(repair_best['checkpoint'],map_location='cpu')
    assert saved['contract']['readout']==best and saved['contract']['data']==source['contract']
    assert readout['contract']['data_contract']==source['contract']
    assert readout['contract']['source_sha256']==saved['contract']['teacher']['sha256']==source['checkpoint']['sha256']
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda();base.shrink_conv.load_state_dict(readout['readout'],strict=True);base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(saved['repair'],strict=True)
    student=copy.deepcopy(base);memory,energy=projected_gru(base.feature_memory,a.rank);student.feature_memory=memory.cuda().eval()
    for m in (base,student,split,repair):
        for p in m.parameters():p.requires_grad_(False)
    trainable=[]
    for name in ('reset_gate_conv','update_gate_conv','candidate_conv'):
        for p in getattr(student.feature_memory.fusion,name).parameters():p.requires_grad_(True);trainable.append(p)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train'];bounds=cfg['model']['args']['lidar_range']
    contract=dict(rank=a.rank,data=source['contract'],source=source['checkpoint'],readout=best,repair=repair_best,
        seed=303,epochs_max=12,timeout_seconds=28800,retained_weight_energy=energy,trainable_parameters=sum(p.numel() for p in trainable),
        frozen='Pillar, all backbone/deblocks/readout/heads, R, post_fusion and BN running statistics. Only three projected gates including BN affine learn.',
        recurrence='Own previous hidden state detached per frame; no teacher forcing, no multi-frame BPTT. Own detached full deep anchors every3 frames/age.25.',
        train='GT + .2 frozen compact base head KD per full/cached view; equal mean on partial frames, one full view on refresh. Add .5 current stage1 feature KD. No augmentation/delay.',
        optimizer='AdamW1e-4 wd1e-4 clip5, full AP plateau factor.5 patience1 threshold.001; min6 epochs then stale4 and lr<=1e-5 stop.',
        selection='Primary best uses full AP; cached AP is reported for that same checkpoint. Preserve epoch0 as fallback. Dense GRU only; no new regional skipping.')
    torch.manual_seed(303);write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',source)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=engineering(base,student,split,repair,cfg,train);write_json(out/'engineering.json',gate)
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();history=[]
    zero=out/'epoch_000';zero.mkdir();path=zero/'memory.pth'
    torch.save(dict(memory=student.feature_memory.state_dict(),contract=contract,epoch=0),str(path))
    report=evaluate(base,student,split,repair,cfg,val,zero,gate['conv_macs'],0)
    expected={32:(.7528362281190099,.7515662092686006),64:(.7554850768747318,.753797275546895)}[a.rank]
    for name,ap in zip(('full','cached'),expected):assert abs(report['results'][name]['ap70']-ap)<1e-8
    best_ap=significant=report['results']['full']['ap70'];stale=0;stop='budget_not_convergence'
    def save_best(ap,report,epoch,path):
        write_json(out/'best.json',dict(ap70=ap,cached_ap70=report['results']['cached']['ap70'],epoch=epoch,checkpoint=str(path),sha256=digest(path)))
    save_best(best_ap,report,0,path)
    for epoch in range(1,13):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();teacher_state=student_state=teacher_anchor=student_anchor=None
        losses=[];depths={0:0,1:0,2:0};tc,th=capture_features(base);sc,sh=capture_features(student)
        _,stream=loader(train,cfg,4,303+epoch)
        try:
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():
                    target,teacher_state,ti=base.step(ego,teacher_state,meta)
                    target_cache,teacher_anchor,tp=cached_view(base,split,repair,tc,teacher_anchor,meta,ti,bounds)
                optimizer.zero_grad(set_to_none=True)
                pred,student_state,si=student.step(ego,student_state,meta)
                pred_cache,student_anchor,sp=cached_view(student,split,repair,sc,student_anchor,meta,si,bounds)
                assert tp==sp and teacher_anchor['stamp']==student_anchor['stamp']
                assert not student_state['value'].requires_grad and not student_anchor['value'].requires_grad
                views=[(pred,target)]+([(pred_cache,target_cache)] if sp else [])
                loss=.5*feature_distillation(sc['first'],tc['first'],target)
                for output,reference in views:
                    term=.2*foreground_head_loss(output,reference)
                    if meta['supervised']:term=term+criterion(output,ego['label_dict'])
                    loss=loss+term/len(views)
                assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(trainable,5.))
                optimizer.step();losses.append(float(loss.detach()));depths[student_anchor['frames']]+=1
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
        finally:
            for hook in th+sh:hook.remove()
        assert index+1==len(train) and all(v>0 for v in depths.values())
        assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
        path=directory/'memory.pth';torch.save(dict(memory=student.feature_memory.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(base,student,split,repair,cfg,val,directory,gate['conv_macs'],epoch)
        ap=report['results']['full']['ap70'];scheduler.step(ap)
        if ap>best_ap:best_ap=ap;save_best(ap,report,epoch,path)
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),depths=depths,lr=optimizer.param_groups[0]['lr'],stale=stale,validation=report))
        write_json(out/'history.json',history)
        torch.save(dict(memory=student.feature_memory.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,contract=contract),str(out/'latest.pth'))
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=1e-5:stop='plateau';break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),stop_reason=stop,best=json.loads((out/'best.json').read_text())))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','readout-run','repair-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--rank',type=int,choices=(32,64),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Projected GRU training exceeded8 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(28800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_projected_gru']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=28800,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

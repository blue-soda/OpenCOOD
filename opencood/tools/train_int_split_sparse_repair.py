"""Bounded mask-aware fine-tuning of the fixed deep-contribution repair.

The heuristic mask is unchanged. Training computes dense R and masks its
output; reported inference runs cropped R, with an equal-mask dense control.
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
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout,SplitReadoutDetector,split_change_request
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_cache_repair import tile_mask,expand_tiles
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.models.sub_modules.int_anchor_residual import refresh_reason,predict
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.train_int_split_repair import capture_teacher,engineering
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def masked_feature(repair,first,cache,meta,bounds):
    old,valid=warp_bev(cache['value'],cache['pose'],meta['pose'],bounds)
    request=split_change_request(early_descriptor(first,4),cache['desc'],cache['pose'],meta['pose'],
        bounds,valid,.5,first.shape[-2:])
    written=expand_tiles(tile_mask(request,4),4,old.shape[-2:])
    candidate=repair(torch.cat([first,old,valid.to(old.dtype)],1))
    return torch.where(written,candidate,old),written


def make_modes(student,split,repair,initial,cfg):
    return dict(full=SplitReadoutDetector(student,cfg,split,1),cached=SplitReadoutDetector(student,cfg,split,3),
        initial=SplitReadoutDetector(student,cfg,split,3,initial,.5,'sparse'),
        repaired=SplitReadoutDetector(student,cfg,split,3,repair,.5,'sparse'),
        masked=SplitReadoutDetector(student,cfg,split,3,repair,.5,'masked_dense'))


def regional_engineering(teacher,student,split,repair,cfg,rows,costs):
    capture,hooks=capture_teacher(teacher);source_state=anchor=None;states=[None,None]
    modes=[SplitReadoutDetector(student,cfg,split,3,repair,.5,e) for e in ('sparse','masked_dense')]
    maximum=0.;partials=0;coverage=set();calls=[0];module_costs={}
    groups=dict(feature_memory=student.feature_memory,stage1=student.backbone.resnet.layer0,
        stage2=student.backbone.resnet.layer1,stage3=student.backbone.resnet.layer2,
        current_decode=student.backbone.deblocks[0],deep_decode2=student.backbone.deblocks[1],deep_decode3=student.backbone.deblocks[2],
        current_projection=split.current,deep_projection=split.history,spatial_readout_first=split.spatial,
        spatial_readout_tail=split.tail,repair=repair,classification_head=student.cls_head,regression_head=student.reg_head)
    if student.use_dir:groups['direction_head']=student.dir_head
    def hit(m,x,y):calls[0]+=1
    hooks += [m.register_forward_hook(hit) for m in (student.backbone.resnet.layer1,student.backbone.resnet.layer2,
        student.backbone.deblocks[1],student.backbone.deblocks[2],split.history)]
    _,stream=loader(rows[:32],cfg,0,303);bounds=cfg['model']['args']['lidar_range']
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                _,source_state,source_info=teacher.step(ego,source_state,meta)
                first=capture['first'];decoded=capture['decoded'];c=split.current_channels
                current=split.current(decoded[:,:c]);target=split.history(decoded[:,c:])
                reason=refresh_reason(anchor,meta,source_info['reset'],3)
                if reason:anchor=dict(value=target,desc=early_descriptor(first,4),pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
                feature=target if reason else masked_feature(repair,first,anchor,meta,bounds)[0]
                expected=predict(student,split.fuse(current,feature))
                if not reason:
                    anchor=dict(anchor,frames=anchor['frames']+1);partials+=1;coverage.add(anchor['frames'])
                    probe=(first,copy.copy(anchor),meta,current)
                for i,mode in enumerate(modes):
                    before=calls[0];previous=states[i]
                    counter=ConvCounter(student,torch.nn.ModuleList([split,repair]))
                    group_counters={k:ConvCounter(m,torch.nn.ModuleList()) for k,m in groups.items()} if i==1 else {}
                    try:pred,states[i],info=mode.step(ego,states[i],meta)
                    finally:
                        counter.close()
                        for group in group_counters.values():group.close()
                    if group_counters:
                        measured={k:g.total for k,g in group_counters.items()}
                        assert sum(measured.values())==counter.total
                        cost_key='partial_masked_dense' if info['local_partial'] else 'full'
                        if cost_key in module_costs:assert module_costs[cost_key]==measured
                        module_costs[cost_key]=measured
                    if info['local_partial']:
                        macs=costs['cached_partial']+(costs['repaired_partial']-costs['cached_partial'])*info['repair_stats']['executed_input_area_ratio']
                        assert calls[0]==before and states[i]['cache']['value'] is previous['cache']['value']
                        assert states[i]['cache']['desc'] is previous['cache']['desc']
                    else:macs=costs['full_full']
                    assert abs(counter.total-macs)<.1
                    for k in pred:
                        maximum=max(maximum,float((pred[k]-expected[k]).abs().max()))
                        assert torch.allclose(pred[k],expected[k],atol=1e-4,rtol=1e-4)
        first,anchor,meta,current=probe
        feature,mask=masked_feature(repair,first,anchor,meta,bounds)
        feature.retain_grad();predict(student,split.fuse(current,feature))['rm'].square().mean().backward()
        assert float(repair.net[-1].weight.grad.abs().sum())>0
        assert all(torch.isfinite(p.grad).all() for p in repair.parameters())
        assert all(p.grad is None for m in (teacher,student,split) for p in m.parameters())
    finally:
        for hook in hooks:hook.remove()
        repair.zero_grad(set_to_none=True)
    assert partials>0 and coverage=={1,2}
    return dict(status='PASSED',frames=32,partial_frames=partials,train_sparse_masked_max_error=maximum,
        immutable_anchors=True,partial_heavy_calls=0,area_cost_hook_parity=True,masked_gradient_checked=True,
        module_conv_macs=module_costs)


def evaluate(student,split,repair,initial,cfg,rows,directory,costs,epoch):
    modes=make_modes(student,split,repair,initial,cfg);states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    timing={k:[] for k in modes};compute={k:0 for k in modes};areas={k:[] for k in modes};maximum=0.
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];preds={}
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                if index>=10:timing[name].append(ms)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if info['local_partial']:
                    area=info['repair_stats']['executed_input_area_ratio'];areas[name].append(area)
                    compute[name]+=costs['cached_partial']+(costs['repaired_partial']-costs['cached_partial'])*area
                else:compute[name]+=costs['full_full']
                if name in ('repaired','masked'):preds[name]=pred
                if meta['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            for k in preds['repaired']:
                p,q=preds['repaired'][k],preds['masked'][k]
                maximum=max(maximum,float((p-q).abs().max()));assert torch.allclose(p,q,atol=1e-4,rtol=1e-4)
            if (index+1)%200==0:write_json(directory.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=index+1))
    assert index+1==len(rows)
    result={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(timing[k])),p95_ms=float(np.percentile(timing[k],95)),
        conv_gflops=2*compute[k]/len(rows)/1e9,partial_frames=len(areas[k]),
        partial_repair_input_area=float(np.mean(areas[k])) if areas[k] else None) for k in modes}
    for k,expected in dict(full=.7562138415770423,cached=.752781663200643,initial=.7532738909256946).items():
        assert abs(result[k]['ap70']-expected)<1e-8
    assert abs(result['repaired']['ap70']-result['masked']['ap70'])<1e-8
    assert len({len(areas[k]) for k in ('cached','initial','repaired','masked')})==1
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=result,
        crop_masked_prediction_max_error=maximum,
        scope='Fixed masks; real cropped inference. Same GPU rotating model time excludes IO/H2D/postprocess. Standard convolution MACs x2 only, including executed halos. Training itself is dense masked.')
    write_json(directory/'report.json',report);return report


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    teacher,cfg,val,provenance=setup(a);best,saved=completed_best(a.readout_run)
    source=Path(a.initial_run);assert json.loads((source/'status.json').read_text())['status']=='COMPLETED'
    initial_best=json.loads((source/'best.json').read_text());assert digest(initial_best['checkpoint'])==initial_best['sha256']
    assert initial_best['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    initial_saved=torch.load(initial_best['checkpoint'],map_location='cpu')
    assert initial_saved['contract']['data']==provenance['contract'] and initial_saved['contract']['readout']==best
    assert saved['contract']['data_contract']==provenance['contract']
    assert saved['contract']['source_sha256']==initial_saved['contract']['teacher']['sha256']==provenance['checkpoint']['sha256']
    student=copy.deepcopy(teacher);student.shrink_conv=compact_shrink(teacher.shrink_conv,96)[0].cuda()
    student.shrink_conv.load_state_dict(saved['readout'],strict=True);student.eval()
    split=SplitCompactReadout(student.shrink_conv,student.backbone.deblocks[0][0].out_channels).cuda().eval()
    torch.manual_seed(303);repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(initial_saved['repair'],strict=True)
    initial=copy.deepcopy(repair)
    for m in (teacher,student,split,initial):
        for p in m.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train'];bounds=cfg['model']['args']['lidar_range']
    contract=dict(data=provenance['contract'],readout=best,teacher=provenance['checkpoint'],initial=initial_best,
        channels=96,width=32,period=3,max_age=.25,seed=303,threshold=.5,tile=4,epochs_max=6,
        trainable_parameters=sum(p.numel() for p in repair.parameters()),
        loss='GT + .2 original INT foreground head KD + .5 normalized feature KD on actual masked repaired contribution.',
        train='Only R trainable. Dense R then fixed mask; no correction outside selected tiles. Current E/spatial readout and source anchors frozen; no recursive writeback.',
        optimizer='AdamW 1e-4 wd1e-4 clip5; plateau factor.5 patience1 threshold.0005; six-epoch bounded warm-start, not a convergence claim.',
        inference='Real crops, same fixed shallow-change threshold .5; uncovered tiles forced. Same-mask dense control at every validation.')
    write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',provenance)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=engineering(teacher,student,split,repair,cfg,train)
    gate['regional']=regional_engineering(teacher,student,split,repair,cfg,train,gate['conv_macs'])
    write_json(out/'engineering.json',gate)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    optimizer=torch.optim.AdamW(repair.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.0005,threshold_mode='abs',min_lr=1e-6)
    # Epoch zero is a real evaluated, retained fallback, never discard a better start.
    zero=out/'epoch_000';zero.mkdir();path=zero/'repair.pth'
    torch.save(dict(repair=repair.state_dict(),contract=contract,epoch=0),str(path))
    report=evaluate(student,split,repair,initial,cfg,val,zero,gate['conv_macs'],0)
    assert abs(report['results']['repaired']['ap70']-.7532738909256946)<1e-8
    best_ap=report['results']['repaired']['ap70'];write_json(out/'best.json',dict(ap70=best_ap,epoch=0,checkpoint=str(path),sha256=digest(path)))
    history=[]
    for epoch in range(1,7):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();state=anchor=None;losses=[];areas=[];depths={1:0,2:0}
        capture,hooks=capture_teacher(teacher);_,stream=loader(train,cfg,4,303+epoch)
        try:
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():
                    target_pred,state,info=teacher.step(ego,state,meta);first=capture['first'];decoded=capture['decoded'];c=split.current_channels
                    current=split.current(decoded[:,:c]);target=split.history(decoded[:,c:])
                    reason=refresh_reason(anchor,meta,info['reset'],3)
                    if reason:anchor=dict(value=target,desc=early_descriptor(first,4),pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
                if not reason:
                    optimizer.zero_grad(set_to_none=True);corrected,mask=masked_feature(repair,first,anchor,meta,bounds)
                    pred=predict(student,split.fuse(current,corrected))
                    loss=.2*foreground_head_loss(pred,target_pred)+.5*feature_distillation(corrected,target,target_pred)
                    if meta['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(repair.parameters(),5.))
                    optimizer.step();losses.append(float(loss.detach()));areas.append(float(mask.float().mean()))
                    anchor=dict(anchor,frames=anchor['frames']+1);depths[anchor['frames']]+=1
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
        finally:
            for hook in hooks:hook.remove()
        assert index+1==len(train) and all(v>0 for v in depths.values())
        assert all(p.grad is None for m in (teacher,student,split,initial) for p in m.parameters())
        path=directory/'repair.pth';torch.save(dict(repair=repair.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(student,split,repair,initial,cfg,val,directory,gate['conv_macs'],epoch)
        ap=report['results']['repaired']['ap70'];scheduler.step(ap)
        if ap>best_ap:
            best_ap=ap;write_json(out/'best.json',dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path)))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),write_area=float(np.mean(areas)),updates_by_depth=depths,lr=optimizer.param_groups[0]['lr'],validation=report))
        write_json(out/'history.json',history)
        torch.save(dict(repair=repair.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,contract=contract),str(out/'latest.pth'))
    write_json(out/'report.json',dict(status='COMPLETED',epochs=6,best=json.loads((out/'best.json').read_text()),stop_reason='bounded_mask_adaptation_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=6))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','readout-run','initial-run','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Sparse adaptation exceeded 6 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(21600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_split_sparse_repair']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=21600,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

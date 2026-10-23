"""Bounded head calibration after pointwise spatial recovery plateaus.

All representation weights stay fixed. Compare full-only and full/cache heads
without adding inference operators or overwriting the recovered baseline.
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
from opencood.models.sub_modules.int_pointwise_spatial_state import ContextualPointwiseStage
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.models.sub_modules.int_head_calibration import (
    detection_heads,enable_only_detection_heads,spatial_head_checkpoint,load_spatial_head_checkpoint)
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_projected_gru import capture_features,cached_view,make_modes,evaluate
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def read(path):return json.loads(Path(path).read_text())


def fixed_best(directory):
    root=Path(directory)
    if read(root/'status.json')['status']!='COMPLETED':raise ValueError('Source must be completed')
    best=read(root/'best.json')
    if digest(best['checkpoint'])!=best['sha256']:raise ValueError('Source SHA mismatch')
    return best,torch.load(best['checkpoint'],map_location='cpu')


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for key,value in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,key,value)
    base,cfg,val,source=setup(a)
    rb,readout=completed_best(a.readout_run);mb,memory=fixed_best(a.memory_run)
    pb,repair_saved=fixed_best(a.repair_run);sb,spatial=fixed_best(a.spatial_run)
    assert mb['sha256']=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    assert pb['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    assert sb['sha256']==a.spatial_sha
    assert spatial['contract']['data']==memory['contract']['data']==repair_saved['contract']['data']==source['contract']
    assert spatial['contract']['source']==source['checkpoint']
    assert spatial['contract']['memory']==mb and spatial['contract']['readout']==rb and spatial['contract']['repair']==pb
    assert readout['contract']['source_sha256']==source['checkpoint']['sha256']
    assert repair_saved['contract']['readout']==rb
    assert spatial['contract']['trainable_parameters']==604032 and 'heads' not in spatial
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda();base.shrink_conv.load_state_dict(readout['readout'],strict=True)
    base.feature_memory=projected_gru(base.feature_memory,32)[0].cuda();base.feature_memory.load_state_dict(memory['memory'],strict=True);base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(repair_saved['repair'],strict=True)
    for module in (base,split,repair):
        for parameter in module.parameters():parameter.requires_grad_(False)
    student=copy.deepcopy(base);stage=ContextualPointwiseStage(base.backbone.resnet.layer0).cuda().eval()
    stage.load_state_dict(spatial['stage'],strict=True);student.backbone.resnet.layer0=stage
    heads=enable_only_detection_heads(student);parameters=list(heads.parameters())
    assert all(heads[name].weight.data_ptr()!=detection_heads(base)[name].weight.data_ptr() for name in heads)
    head_prefixes=tuple(name+'.' for name in heads)
    fixed={key:value.detach().cpu().clone() for key,value in student.state_dict().items() if not key.startswith(head_prefixes)}
    args=read(Path(a.root)/'full_gru'/'effective_runtime.json')['arguments']
    train=read(args['manifest'])['train'];bounds=cfg['model']['args']['lidar_range']
    contract=dict(source=source['checkpoint'],data=source['contract'],memory=mb,readout=rb,repair=pb,spatial=sb,
        source_contract=spatial['contract'],views=a.views,seed=303,trainable_parameters=sum(p.numel() for p in parameters),
        trainable='Existing cls_head, reg_head, dir_head only. All feature/memory/BN/readout/repair weights and buffers frozen.',
        objective='GT when supervised + .2 frozen rank32 teacher head KD. Full view only, or equal mean full/cached losses on partial frames. No feature loss because features are fixed.',
        recurrence='Own frozen FM state and detached own deep anchors every3 frames. Spatial H/valid remain zero; no new regional reuse.',
        schedule='New AdamW1e-5 wd1e-4 clip5; fullAP plateau factor.5 patience1 threshold.001; min4 then stale4/lr<=1e-5; max6epochs/4hours.',
        selection='Full AP selects best; cached AP from same checkpoint. Preserve source-equivalent e0. New checkpoint includes both stage and heads; load_spatial_head_checkpoint is required.',
        checkpoint_format='pointwise_stage_and_detection_heads_v1')
    write_json(out/'contract.json',contract);write_json(out/'provenance.json',source)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    # Verify the artifact includes exactly the represented model, including heads.
    restored=copy.deepcopy(student);load_spatial_head_checkpoint(restored,spatial_head_checkpoint(student,contract,0))
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in student.state_dict().items())
    del restored
    modes=make_modes(base,student,split,repair,cfg);states={name:None for name in modes};costs={}
    captured,hooks=capture_features(student);check_state=check_anchor=None;maximum=0.
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'))
            full_reference,check_state,check_info=student.step(ego,check_state,sample['meta'])
            cache_reference,check_anchor,_=cached_view(student,split,repair,captured,check_anchor,sample['meta'],check_info,bounds)
            for name,mode in modes.items():
                counter=ConvCounter(base if name.startswith('base') else student,torch.nn.ModuleList([split,repair]))
                try:pred,states[name],info=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                assert all(torch.isfinite(value).all() for value in pred.values())
                key=name+('_partial' if info['local_partial'] else '_full')
                if key in costs:assert costs[key]==counter.total
                costs[key]=counter.total
                if name in ('full','cached'):
                    reference=full_reference if name=='full' else cache_reference
                    for key,value in pred.items():
                        maximum=max(maximum,float((value-reference[key]).abs().max()))
                        assert torch.allclose(value,reference[key],atol=1e-4,rtol=1e-4)
    for hook in hooks:hook.remove()
    probe_meta=dict(sample['meta'],timestamp_us=int(sample['meta']['timestamp_us'])+100000)
    probe,_,_=student.step(ego,check_state,probe_meta)
    sum(value.square().mean() for value in probe.values()).backward()
    assert all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad)
    assert all(head.weight.grad is not None and torch.isfinite(head.weight.grad).all() and float(head.weight.grad.abs().sum())>0 for head in heads.values())
    student.zero_grad(set_to_none=True)
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,conv_macs=costs,complete_artifact_round_trip=True,
        train_deploy_max_error=maximum,head_gradients=True,frozen_representation_gradients=True))
    expected=(.7567024378706645,.7555532616222492)
    scope='Existing-head calibration on fixed recovered pointwise features. No inference operators added; spatial H remains zero. Own FM/deep anchors. Full/cache AP paired; conv MACsx2 and model-only latency.'
    def assess(epoch):
        folder=out/('epoch_%03d'%epoch);folder.mkdir()
        path=folder/'model.pth';torch.save(spatial_head_checkpoint(student,contract,epoch),str(path))
        report=evaluate(base,student,split,repair,cfg,val,folder,costs,epoch,expected,scope)
        return report,path
    def save_best(report,path,epoch):
        write_json(out/'best.json',dict(epoch=epoch,ap70=report['results']['full']['ap70'],cached_ap70=report['results']['cached']['ap70'],
            checkpoint=str(path),sha256=digest(path),format=contract['checkpoint_format']))
    report,path=assess(0)
    assert abs(report['results']['full']['ap70']-sb['ap70'])<1e-5
    assert abs(report['results']['cached']['ap70']-sb['cached_ap70'])<1e-5
    save_best(report,path,0);best=significant=report['results']['full']['ap70'];stale=0;history=[]
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();stop='budget_not_convergence'
    torch.manual_seed(303)
    for epoch in range(1,7):
        ts=ss=anchor=None;losses=[];captured,hooks=capture_features(student)
        _,stream=loader(train,cfg,4,303+epoch)
        try:
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():target,ts,_=base.step(ego,ts,meta)
                optimizer.zero_grad(set_to_none=True);pred,ss,info=student.step(ego,ss,meta)
                def objective(value):
                    loss=.2*foreground_head_loss(value,target)
                    return loss+criterion(value,ego['label_dict']) if meta['supervised'] else loss
                loss=objective(pred)
                if a.views=='paired':
                    cached,anchor,partial=cached_view(student,split,repair,captured,anchor,meta,info,bounds)
                    if partial:loss=.5*(loss+objective(cached))
                assert torch.isfinite(loss);loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,5.))
                optimizer.step();losses.append(float(loss.detach()))
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
        finally:
            for hook in hooks:hook.remove()
        assert index+1==len(train)
        assert all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad)
        assert all(torch.equal(student.state_dict()[key].cpu(),value) for key,value in fixed.items()),'Frozen representation changed'
        report,path=assess(epoch);ap=report['results']['full']['ap70'];scheduler.step(ap)
        if ap>best:best=ap;save_best(report,path,epoch)
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr'],stale=stale,validation=report))
        write_json(out/'history.json',history)
        latest=spatial_head_checkpoint(student,contract,epoch)
        latest.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,significant=significant,stale=stale)
        torch.save(latest,str(out/'latest.pth'))
        if epoch>=4 and stale>=4 and optimizer.param_groups[0]['lr']<=1e-5:stop='plateau';break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),stop_reason=stop,best=read(out/'best.json')))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','readout-run','repair-run','memory-run','spatial-run','spatial-sha','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--views',choices=('full','paired'),required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--worker',action='store_true')
    a=parser.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Head calibration exceeded4 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(14400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_pointwise_heads']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=14400,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

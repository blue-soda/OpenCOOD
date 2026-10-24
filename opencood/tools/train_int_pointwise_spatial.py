"""Dense precision recovery before spatial-state reuse: frozen mature IO path."""
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
from opencood.models.sub_modules.int_pointwise_spatial_state import ContextualPointwiseStage,ExpandedContextPointwiseStage
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_short_spatial_state import ShortSpatialStage
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_projected_gru import capture_features,make_modes,evaluate
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.int_spatial_resume import load_completed_spatial_resume


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    base,cfg,val,source=setup(a);readout_best,readout=completed_best(a.readout_run)
    memory_root=Path(a.memory_run);repair_root=Path(a.repair_run)
    assert json.loads((memory_root/'status.json').read_text())['status']=='COMPLETED'
    assert json.loads((repair_root/'status.json').read_text())['status']=='COMPLETED'
    memory_best=json.loads((memory_root/'best.json').read_text());repair_best=json.loads((repair_root/'best.json').read_text())
    assert digest(memory_best['checkpoint'])==memory_best['sha256']=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    assert digest(repair_best['checkpoint'])==repair_best['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    saved=torch.load(memory_best['checkpoint'],map_location='cpu');r=torch.load(repair_best['checkpoint'],map_location='cpu')
    assert saved['contract']['data']==r['contract']['data']==source['contract']
    assert saved['contract']['source']['sha256']==readout['contract']['source_sha256']==source['checkpoint']['sha256']
    assert r['contract']['readout']==readout_best
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda();base.shrink_conv.load_state_dict(readout['readout'],strict=True)
    base.feature_memory=projected_gru(base.feature_memory,32)[0].cuda();base.feature_memory.load_state_dict(saved['memory'],strict=True);base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(r['repair'],strict=True)
    for m in (base,split,repair):
        for p in m.parameters():p.requires_grad_(False)
    torch.manual_seed(303);student=copy.deepcopy(base)
    stage=(ShortSpatialStage(base.backbone.resnet.layer0,context=a.short_spatial=='context') if a.short_spatial else
           ProjectedSpatialStage(base.backbone.resnet.layer0,a.projected_rank) if a.projected_rank
           else ContextualPointwiseStage(base.backbone.resnet.layer0)).cuda()
    student.backbone.resnet.layer0=stage
    student.eval();parameters=[p for p in stage.parameters() if p.requires_grad]
    assert parameters and all(not p.requires_grad for p in stage.stem.parameters())
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train']
    contract=dict(source=source['checkpoint'],data=source['contract'],memory=memory_best,readout=readout_best,repair=repair_best,
        architecture='Frozen mature first block -> dense cheap depthwise3/9 context -> pointwise513to512to512to128 update plus current stem residual. Frozen e11 GRU, mature downstream backbone/readout/heads.',
        trainable_parameters=sum(p.numel() for p in parameters),phase='A_dense_current_only; old spatial H and valid are zeros, no region reuse',
        objective='GT when supervised + .2 head KD +1.0 first-stage feature KD. Own e11 FM state, detached each frame. No cached-view loss in phase A; report its compatibility separately.',
        schedule='AdamW3e-4 wd1e-4 clip5; fullAP plateau factor.5 patience1 threshold.001; min6 then stale4 and lr<=1e-5; max12epochs/8hours.',
        selection='Full AP primary best; cached AP from same checkpoint. Epoch0 retained. Precision recovery, not a sparse-compute result.')
    if a.projected_rank:
        contract.update(architecture='Frozen mature stem and original BN/residual/ReLU. Four 3x3 kernels replaced by SVD-initialized 1x1 input-channel projection128to%d then 3x3 %dto128 without intermediate nonlinearity. Four original post-activation map shapes retained.'%(a.projected_rank,a.projected_rank),
            rank=a.projected_rank,retained_weight_energy=stage.retained_weight_energy,
            phase='Dense precision recovery for projected four-map spatial update; no new spatial cache during training/validation',
            schedule='AdamW1e-4 wd1e-4 clip5; fullAP plateau factor.5 patience1 threshold.001; fixed first4epochs/8hours diagnostic budget, not convergence. Reassess curve before extension.',
            trainable_scope='Only four new projection/spatial kernel pairs. Stem, all BN including affine/statistics, FM, downstream/readout/repair/heads frozen.',
            objective='GT when supervised + .2 head KD +1.0 first-stage feature KD. Mature frozen teacher uses train split only for optimization. Own frozen e11 FM state detached each frame. Full and original deep-cache validation; no teacher state injection.')
    warm_start=bool(a.warm_start)
    if a.short_spatial:
        contract.update(architecture='Frozen mature stem -> optional DW5+PW1 zero-residual current-context -> ONE mature128-channel residual block with TWO3x3 layers -> unchanged mature downstream/readout/heads. Deletes original last residual block; both arms initialized to same first-block function. Expensive local updater receptive field5; context arm total nominal9 including cheap context.',
            short_spatial=a.short_spatial,
            phase='Dense precision gate for two-layer single-H local updater; NO new spatial reuse during training/validation',
            objective='GT when supervised +.2 head KD +1.0 normalized foreground/global H feature KD from frozen mature model on train split only. Teacher/student own FM state detached each frame; no teacher state injection. Full validation and original deep-period3 compatibility reported independently.',
            schedule='AdamW1e-4 wd1e-4 clip5; fullAP plateau factor.5 patience1 threshold.001; first4epochs/8hours diagnostic budget, not convergence. Review learning curves before extension or local caching.',
            trainable_scope='Only retained block two3x3 weights plus optional context DW5/PW1 weights. Stem, all BN including affine/statistics, FM, downstream/readout/repair/heads frozen. Context PW starts zero, DW identity. No architecture selection based on sparse AP yet.')
    if warm_start:
        assert a.warm_start_sha
        assert a.warm_start_lr>0 and a.stem_lr>0
        if a.adapt_stem and a.expand_context:
            raise ValueError('Compare stem adaptation and context expansion separately')
        assert digest(a.warm_start)==a.warm_start_sha
        warm=torch.load(a.warm_start,map_location='cpu')
        parent_contract=dict(warm['contract']);parent_contract.pop('resume',None)
        assert parent_contract==contract,'Warm-start must share the unchanged dense recovery contract'
        stage.load_state_dict(warm['stage'],strict=True)
        warm_validation=json.loads((Path(a.warm_start).parent/'report.json').read_text())
        assert warm_validation['status']=='COMPLETED'
        if a.expand_context:
            stage=ExpandedContextPointwiseStage.from_stage(stage)
            student.backbone.resnet.layer0=stage
        if a.adapt_stem:stage.enable_stem_adaptation()
        parameters=[p for p in stage.parameters() if p.requires_grad]
        contract['warm_start']=dict(checkpoint=a.warm_start,sha256=a.warm_start_sha,parent_epoch=warm['epoch'],
            expanded_context=a.expand_context,adapt_stem=a.adapt_stem,
            optimizer='New AdamW for both paired arms, not a continuation')
        contract['trainable_parameters']=sum(p.numel() for p in parameters)
        if a.expand_context:
            contract['architecture']+=' Extra two directional-initialized depthwise3 responses per stem channel; U769to512to512to128. Zero new mixing columns preserve initial output.'
        if a.adapt_stem:
            contract['architecture']=contract['architecture'].replace('Frozen mature first block','Adaptable mature first block with fixed BN statistics')
            contract['architecture']+=' Mature first-block weights and BN affine adapt at a lower learning rate; BN running statistics remain fixed. No extra parameters or inference operators.'
        contract['schedule']='Paired warm-start diagnostic: AdamW stage_lr=%g stem_lr=%s wd1e-4 clip5; same plateau scheduler; max6epochs/8hours. Source optimizer deliberately not restored in either arm.'%(a.warm_start_lr,str(a.stem_lr) if a.adapt_stem else 'frozen')
    elif a.expand_context or a.warm_start_sha or a.adapt_stem:
        raise ValueError('Architecture diagnostics require an explicit fixed warm-start checkpoint and SHA')
    phase_epochs_max=4 if (a.projected_rank or a.short_spatial) else (6 if warm_start else 12)
    resume=load_completed_spatial_resume(a.resume_run,contract,phase_epochs_max) if a.resume_run else None
    if resume:
        stage.load_state_dict(resume['saved']['stage'],strict=True)
        contract['resume']=resume['provenance']
    write_json(out/'contract.json',contract);write_json(out/'provenance.json',source)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    modes=make_modes(base,student,split,repair,cfg);states={k:None for k in modes};costs={}
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'))
            for name,mode in modes.items():
                root=base if name.startswith('base') else student
                counter=ConvCounter(root,torch.nn.ModuleList([split,repair]))
                try:pred,states[name],info=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                assert all(torch.isfinite(v).all() for v in pred.values())
                key=name+('_partial' if info['local_partial'] else '_full')
                if key in costs:assert costs[key]==counter.total
                costs[key]=counter.total
    gradient_check=None
    if a.projected_rank or a.short_spatial:
        initial={k:v.detach().clone() for k,v in stage.state_dict().items()}
        tc,th=capture_features(base);sc,sh=capture_features(student)
        temporary=torch.optim.SGD(parameters,lr=1e-4)
        try:
            stage.train()
            with torch.no_grad(): target,_,_=base.step(ego,None,sample['meta'])
            pred,_,_=student.step(ego,None,sample['meta'])
            loss=feature_distillation(sc['first'],tc['first'],target)+.2*foreground_head_loss(pred,target)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
            assert torch.isfinite(norm) and float(norm)>0
            temporary.step()
            assert any(not torch.equal(initial[n],p) for n,p in stage.named_parameters() if p.requires_grad)
            assert all(torch.equal(initial[n],p) for n,p in stage.named_parameters() if not p.requires_grad)
            assert all(torch.equal(initial[n],b) for n,b in stage.named_buffers())
            assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
            gradient_check=dict(loss=float(loss.detach()),gradient_norm=float(norm),only_factors_updated=True,restored_before_epoch0=True)
        finally:
            for hook in th+sh:hook.remove()
            stage.load_state_dict(initial,strict=True);stage.zero_grad(set_to_none=True);stage.eval()
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,conv_macs=costs,gradient_check=gradient_check))
    expected=(.7567024378706645,.7555532616222492)
    scope='Dense pointwise spatial baseline with frozen e11 GRU. New spatial history input is zero; no regional reuse. Own FM states/deep anchors, paired full and cached validation. Standard conv MACsx2, model latency excludes IO/H2D/postprocess.'
    if a.projected_rank:
        scope='Dense projected spatial rank%d recovery; unchanged mature stem/BN/residuals and four-map state interface. No new spatial reuse enabled. Own FM and original deep anchors, full and cached validation. Conv MACsx2; no inference teacher. Model timing excludes IO/H2D/postprocess.'%a.projected_rank
    if a.short_spatial:
        scope='Dense short spatial updater arm '+a.short_spatial+': frozen mature stem, one retained two-convolution block with optional cheap current DW5/PW1 context. Own FM states; full and original deep-period3 compatibility. No spatial reuse or inference teacher. Conv MACsx2; model timing excludes IO/H2D/postprocess. Precision gate before local reuse, not convergence proof.'
    def assess(epoch):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();stage.eval()
        path=directory/'stage.pth';torch.save(dict(stage=stage.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(base,student,split,repair,cfg,val,directory,costs,epoch,expected,scope)
        return report,path
    def save_best(report,path,epoch):
        write_json(out/'best.json',dict(epoch=epoch,ap70=report['results']['full']['ap70'],cached_ap70=report['results']['cached']['ap70'],checkpoint=str(path),sha256=digest(path)))
    if warm_start and a.adapt_stem:
        stem_parameters=list(stage.stem.parameters());stem_ids={id(p) for p in stem_parameters}
        groups=[dict(params=[p for p in parameters if id(p) not in stem_ids],lr=a.warm_start_lr),
                dict(params=stem_parameters,lr=a.stem_lr)]
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    else:
        optimizer=torch.optim.AdamW(parameters,lr=a.warm_start_lr if warm_start else (1e-4 if (a.projected_rank or a.short_spatial) else 3e-4),weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-6)
    if resume:
        optimizer.load_state_dict(resume['saved']['optimizer']);scheduler.load_state_dict(resume['saved']['scheduler'])
        history=resume['saved']['history'];start_epoch=resume['epoch']
        best=resume['best']['ap70'];significant=resume['significant'];stale=resume['stale']
        write_json(out/'best.json',resume['best']);write_json(out/'history.json',history)
        write_json(out/'resume.json',resume['provenance'])
        # Engineering probes run before this point. Restore RNGs afterwards so
        # those probes cannot perturb epoch-boundary continuation.
        torch.set_rng_state(resume['saved']['torch_rng'])
        torch.cuda.set_rng_state_all(resume['saved']['cuda_rng'])
        np.random.set_state(resume['saved']['numpy_rng'])
    else:
        report,path=assess(0);best=significant=report['results']['full']['ap70'];stale=0
        if warm_start:
            for name in ('full','cached'):
                assert abs(report['results'][name]['ap70']-warm_validation['results'][name]['ap70'])<1e-5,'Warm-start AP changed before training'
        save_best(report,path,0);history=[];start_epoch=0
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();stop='budget_not_convergence'
    for epoch in range(start_epoch+1,start_epoch+phase_epochs_max+1):
        stage.train();ts=ss=None;losses=[];tc,th=capture_features(base);sc,sh=capture_features(student)
        _,stream=loader(train,cfg,4,303+epoch)
        try:
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():target,ts,_=base.step(ego,ts,meta)
                optimizer.zero_grad(set_to_none=True);pred,ss,_=student.step(ego,ss,meta)
                loss=feature_distillation(sc['first'],tc['first'],target)+.2*foreground_head_loss(pred,target)
                if meta['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,5.))
                optimizer.step();losses.append(float(loss.detach()))
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
        finally:
            for hook in th+sh:hook.remove()
        assert index+1==len(train) and all(p.grad is None for p in student.parameters() if not p.requires_grad)
        report,path=assess(epoch);ap=report['results']['full']['ap70'];scheduler.step(ap)
        if ap>best:best=ap;save_best(report,path,epoch)
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr'],stale=stale,validation=report))
        if warm_start and a.adapt_stem:history[-1]['stem_lr']=optimizer.param_groups[1]['lr']
        write_json(out/'history.json',history)
        torch.save(dict(stage=stage.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,contract=contract,
            epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            significant=significant,stale=stale),str(out/'latest.tmp'))
        os.replace(str(out/'latest.tmp'),str(out/'latest.pth'))
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=1e-5:stop='plateau';break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),phase_epochs=len(history)-start_epoch,
        stop_reason=stop,best=json.loads((out/'best.json').read_text())))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','readout-run','repair-run','memory-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--resume-run',help='Completed budget-limited spatial run; continue in a new output directory')
    p.add_argument('--warm-start',help='Immutable stage checkpoint for a separate paired six-epoch capacity diagnostic')
    p.add_argument('--warm-start-sha')
    p.add_argument('--expand-context',action='store_true')
    p.add_argument('--adapt-stem',action='store_true',help='Warm-start only: unfreeze mature stem weights but keep BN statistics fixed')
    p.add_argument('--warm-start-lr',type=float,default=1e-4)
    p.add_argument('--stem-lr',type=float,default=1e-5)
    p.add_argument('--projected-rank',type=int,choices=(64,96),help='Separate four-epoch SVD input-channel compression recovery diagnostic')
    p.add_argument('--short-spatial',choices=('plain','context'),help='Four-epoch dense precision gate for two-layer spatial update')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.short_spatial and (a.projected_rank or a.warm_start or a.expand_context or a.adapt_stem or a.warm_start_sha):
        p.error('Short spatial gate starts from fixed mature model, without other architecture/warm-start options')
    if a.projected_rank and (a.resume_run or a.warm_start or a.expand_context or a.adapt_stem or a.warm_start_sha):
        p.error('Projected spatial recovery starts from fixed mature kernels, not an old pointwise checkpoint')
    if a.worker:
        def expired(s,f):raise TimeoutError('Pointwise spatial recovery exceeded8 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(28800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_pointwise_spatial']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=28800,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()

"""Train only the cheap complete readout-state updater, fixed mature backbone."""
import copy
import json
import numpy as np
import torch
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.models.sub_modules.int_short_readout import ShortReadoutState
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_anchor_residual import predict
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of,evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_gpu_exclusivity import sample_gpu_usage


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,str(temporary));temporary.replace(path)


def train_short_readout(a,cfg,train,val,baseline,out,costs,spatial_macs,deep_repair,checkpoint,source,expected):
    torch.manual_seed(303)
    control=RecurrentDeepDetector(baseline,deep_repair,4)
    mode=RecurrentDeepDetector(baseline,deep_repair,4)
    kind=a.short_readout_kind
    if kind=='residual':repair=LightRepair(256,a.short_readout_width,current_channels=96).cuda().eval()
    else:
        repair,initialization=current_readout_from_mature(mode.readout,a.short_readout_width,kind=='current_gated')
        repair=repair.cuda().eval()
    cache=ShortReadoutState(mode.readout,repair,mode.bounds,3);mode.readout_cache=cache
    contract=dict(source={k:source[k] for k in ('source','memory','readout','repair','basis','data')},
        frozen_deep_checkpoint=checkpoint,width=a.short_readout_width,seed=303,deep_period=4,readout_period=3,
        epochs_max=a.short_readout_epochs,tbptt=2,trainable_parameters=sum(p.numel() for p in repair.parameters()),
        architecture='Current96 + aligned complete readout256 + valid1 ->1x1(width) -> depthwise3x3 ->1x1(256), residual added to old complete readout. Complete output written every frame.',
        inference='Readout full only period3/.25s/reset; dense cheap correction on ordinary frames. Fixed four-layer H locality, low-rank FM and frozen four-frame D. No local short-readout skipping yet.',
        teacher='Original full readout on SAME current input only during training; discarded on ordinary frames, never injected into state. Own H confidence from student predictions.',
        loss='GT labelled ordinary frames +.2 foreground head KD +.5 feature KD; only short readout updater trainable.',
        optimization='AdamW1e-4 wd1e-4 clip5; Plateau .5 patience1 threshold.0005 minlr1e-6. Min6 epochs, stale4 and lr<=2.5e-5. Maximum budget does not imply convergence.',
        purpose='Check precision of cheap complete-state path before localizing it. Width32/64 planned capacity controls; retain validated dense/local readout baseline.')
    if kind!='residual':
        contract.update(kind=kind,initialization=initialization,
            architecture='Current96 ->3x3(width)/ReLU ->3x3(width) ->1x1(256)/ReLU; mature BN-folded and rank-factor initialized. '+
            ('Bounded per-cell history blend g*old+(1-g)*fresh; g=.95sigmoid(current,old_mean,abs_difference_mean,valid)*valid.' if kind=='current_gated' else 'Current-only control; ordinary output ignores history.'),
            purpose='Test stronger current spatial expression and bounded history reuse before localizing; complete output state, same data/loss/periods.')
    write_json(out/'short_readout_contract.json',contract)
    write_json(out/'validation_execution.json',dict(contention='User permits shared GPU training and validation. Sample other processes without waiting or exiting for sharing; any observed sharing invalidates whole evaluation latency, not AP. Hardware/query/CUDA allocation errors still propagate.',
        scope='Execution only; model/training contract and strict standalone benchmarks unchanged.'))
    states={'control':None,'student':None};samples=[];partial=0;maximum=0.
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];previous=states['student']
            if len(samples)<6:samples.append((ego,meta))
            counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair,repair]))
            try:pred,states['student'],info=mode.step(ego,previous,meta)
            finally:counter.close()
            assert counter.total+info['spatial_extra_linear_macs']==cost_of(costs,spatial_macs,info,True)
            cache.training_targets=True
            other,other_state,_=mode.step(ego,previous,meta);cache.training_targets=False
            for k in pred:assert torch.equal(pred[k],other[k])
            assert torch.equal(states['student']['readout']['value'],other_state['readout']['value'])
            partial+=int(not info['readout_refresh'])
            old_cache=mode.readout_cache;one=ShortReadoutState(mode.readout,repair,mode.bounds,1);mode.readout_cache=one
            try:left,_,_=mode.step(ego,None if states['control'] is None else dict(states['control'],readout=None),meta)
            finally:mode.readout_cache=old_cache
            right,states['control'],_=control.step(ego,states['control'],meta)
            for k in left:maximum=max(maximum,float((left[k]-right[k]).abs().max()));assert torch.equal(left[k],right[k])
    cache.training_targets=True;state=None;terms=[]
    for ego,meta in samples:
        pred,state,info=mode.step(ego,state,meta)
        if not info['readout_refresh']:terms.append(pred['rm'].square().mean())
    assert partial and terms
    torch.stack(terms).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in repair.parameters())
    assert any(float(p.grad.abs().sum())>0 for p in repair.parameters())
    assert all(p.grad is None for module in (mode.model,mode.readout,mode.repair) for p in module.parameters())
    repair.zero_grad(set_to_none=True);cache.training_targets=False
    write_json(out/'short_readout_engineering.json',dict(status='PASSED',frames=32,partial_frames=partial,
        period1_max_error=maximum,teacher_on_off_exact=True,actual_macs_verified=True,gradient_only_short_updater=True))
    optimizer=torch.optim.AdamW(repair.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.0005,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();history=[];best=None;significant=-1.;stale=0;start=0;pending_eval=None
    if a.short_readout_resume:
        saved=torch.load(a.short_readout_resume,map_location='cpu')
        assert dict(saved['contract'],epochs_max=a.short_readout_epochs)==contract
        repair.load_state_dict(saved['repair'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];significant=saved['significant'];stale=saved['stale']
        start=saved['epoch'] if saved['next_action']=='evaluate' else saved['epoch']+1
        pending_eval=saved.get('training_summary') if saved['next_action']=='evaluate' else None
        assert start<=a.short_readout_epochs
        write_json(out/'resume.json',dict(path=a.short_readout_resume,sha256=digest(a.short_readout_resume),next_epoch=start,next_action=saved['next_action']))
    for epoch in range(start,a.short_readout_epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();losses=[];updates=0
        if pending_eval is not None:losses=pending_eval['losses'];updates=pending_eval['updates'];pending_eval=None
        elif epoch:
            cache.training_targets=True;state=None;terms=[];optimizer.zero_grad(set_to_none=True)
            _,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));pred,state,info=mode.step(ego,state,sample['meta'])
                if not info['readout_refresh']:
                    with torch.no_grad():teacher=predict(mode.model,info['readout_target'])
                    loss=.2*foreground_head_loss(pred,teacher)+.5*feature_distillation(info['readout_student'],info['readout_target'],teacher)
                    if sample['meta']['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);terms.append(loss);losses.append(float(loss.detach()))
                if len(terms)>=2 or info['readout_refresh'] or index+1==len(train):
                    if terms:
                        torch.stack(terms).mean().backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(repair.parameters(),5.))
                        optimizer.step();optimizer.zero_grad(set_to_none=True);terms=[];updates+=1
                    state['readout']['value']=state['readout']['value'].detach()
                if (index+1)%200==0:
                    torch.cuda.synchronize()
                    evidence=sample_gpu_usage(a.gpu)
                    if not evidence['timing_valid']:
                        write_json(directory/('train_shared_%06d.json'%(index+1)),evidence)
                    write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
            assert index+1==len(train) and losses and not terms
            assert all(p.grad is None for module in (mode.model,mode.readout,mode.repair) for p in module.parameters())
        cache.training_targets=False;path=directory/'repair.pth'
        atomic_save(dict(repair=repair.state_dict(),contract=contract,epoch=epoch),path)
        def snapshot(action):return dict(repair=repair.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,contract=contract,best=best,significant=significant,stale=stale,epoch=epoch,next_action=action)
        atomic_save(dict(snapshot('evaluate'),training_summary=dict(losses=losses,updates=updates)),out/'pending_validation.pth')
        report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,epoch,
            variants=dict(dense_recurrent=control,short_readout=mode),expected_ap={'dense_recurrent':expected},share_gpu=True,
            note='Full5794/1738 independent baseline/dense readout/short readout histories. Full readout teacher disabled in ordinary inference. Epoch0 initialization only. Conv MACsx2; all alignment included in model time.')
        ap=report['results']['short_readout']['ap70']
        if epoch:scheduler.step(ap)
        if best is None or ap>best['ap70']:
            best=dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path));write_json(out/'best.json',best)
        if ap>significant+.0005:significant=ap;stale=0
        elif epoch:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,updates=updates,validation=report,lr=optimizer.param_groups[0]['lr'],stale=stale))
        write_json(out/'history.json',history);atomic_save(snapshot('train'),out/'latest.pth')
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=2.5e-5:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=epoch,best=best,stop_reason='plateau' if epoch<a.short_readout_epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=epoch))

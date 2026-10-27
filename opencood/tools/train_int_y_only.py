"""Train complete-Y recurrence with a frozen cheap front and no H/D state."""
import json
import numpy as np
import torch
from opencood.models.sub_modules.int_y_only import YOnlyUpdater,YOnlyDetector
from opencood.models.sub_modules.int_anchor_residual import predict
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import evaluate
from opencood.tools.train_int_short_readout import atomic_save
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_gpu_exclusivity import sample_gpu_usage


def engineering(mode,full,cfg,train):
    state=full_state=native_state=None;samples=[];ordinary=0;maximum=0.;costs={}
    _,stream=loader(train[:32],cfg,0,303)
    watched=[*(m for m in mode.stage.blocks.modules() if isinstance(m,torch.nn.Conv2d)),mode.model.backbone.resnet.layer1,
             mode.model.backbone.resnet.layer2,*mode.model.backbone.deblocks,
             mode.readout.current,mode.readout.history,mode.readout.spatial,mode.readout.tail]
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            if len(samples)<6:samples.append((ego,meta))
            previous=state;called=[]
            hooks=[module.register_forward_hook(lambda m,x,y:called.append(type(m).__name__)) for module in watched]
            counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.updater]))
            try:pred,state,info=mode.step(ego,previous,meta)
            finally:
                counter.close()
                for hook in hooks:hook.remove()
            refresh=info['y_refresh']
            if refresh in costs:assert costs[refresh]==counter.total
            costs[refresh]=counter.total
            assert set(state)=={'fm','y'}
            assert set(state['y'])=={'value','pose','stamp','scene','frames','full_stamp'}
            if not refresh:
                ordinary+=1;assert not called, 'Ordinary inference executed an H/D/readout module'
            else:assert called
            # Teacher is computed only in this discarded supervised replay.
            other,other_state,_=mode.step(ego,previous,meta,supervise=True)
            for key in pred:assert torch.equal(pred[key],other[key])
            assert torch.equal(state['y']['value'],other_state['y']['value'])
            full_counter=ConvCounter(full.model,torch.nn.ModuleList([full.readout]))
            try:reference,full_state,full_info=full.step(ego,full_state,meta)
            finally:full_counter.close()
            assert full_info['y_refresh'] and full_counter.total==costs[True]
            native,native_state,_=mode.model.step(ego,native_state,meta)
            for key in native:
                error=float((native[key]-reference[key]).abs().max());maximum=max(maximum,error)
                assert torch.allclose(native[key],reference[key],atol=1e-3,rtol=1e-3)
    assert ordinary and set(costs)=={False,True} and costs[False]<costs[True]
    mode.execution_costs=costs;full.execution_costs={True:costs[True]}
    mode.updater.zero_grad(set_to_none=True);state=None;terms=[]
    for ego,meta in samples:
        pred,state,info=mode.step(ego,state,meta,supervise=True)
        if not info['y_refresh']:terms.append(pred['rm'].square().mean())
    assert terms
    torch.stack(terms).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mode.updater.parameters())
    assert any(float(p.grad.abs().sum())>0 for p in mode.updater.parameters())
    assert all(p.grad is None for m in (mode.model,mode.readout) for p in m.parameters())
    mode.updater.zero_grad(set_to_none=True)
    return dict(status='PASSED',frames=32,ordinary_frames=ordinary,ordinary_h_d_readout_calls=0,
        teacher_on_off_exact=True,only_fm_and_y_persist=True,gradient_only_updater=True,
        full_vs_native_prediction_max_error=maximum,
        full_conv_macs=costs[True],ordinary_conv_macs=costs[False],
        cost_scope='Actual Conv/ConvTranspose hooks, same established MACx2 convention; excludes nonconv arithmetic and Pillar pointwise Linear as in existing baseline.')


def train_y_only(a,cfg,train,val,baseline,out,costs,spatial_macs,deep_repair,checkpoint,source,expected):
    torch.manual_seed(303)
    updater=YOnlyUpdater(width=a.y_only_width).cuda().eval()
    mode=YOnlyDetector(baseline,updater,a.y_only_period)
    full=YOnlyDetector(baseline,updater,1)
    contract=dict(kind='y_only_spatial_v1',source={k:source[k] for k in ('source','memory','readout','repair','basis','data')},
        width=a.y_only_width,period=a.y_only_period,max_age_us=mode.cache.max_age_us,seed=303,tbptt=2,
        epochs_max=a.y_only_epochs,trainable_parameters=sum(p.numel() for p in updater.parameters()),
        front='Frozen Pillar/scatter -> lowrank GRU32 (own FM state) -> mature stem first residual block ->128-channel E. No four-layer U, H, D, deblocks or mature readout on ordinary inference.',
        update='E128 + aligned completeY256 + valid1 ->1x1(w)/ReLU ->3x3(w)/ReLU ->3x3(w)/ReLU ->candidate1x1(256)/ReLU. Separate1x1 gate: .95sigmoid*valid; output g*old+(1-g)*candidate. Entire grid initially, local5x5 dependencies.',
        initialization='Kaiming hidden layers, candidate weight N(0,.001), zero candidate bias, zero gate weight/bias4. Near-warp start, not transferred mature Y predictor. Bounded direct old coefficient is not a contraction proof.',
        state='Only completeY plus pose/time/refresh metadata outside FM; no H/D/reference/confidence state. FM retained explicitly. Write every ordinary prediction back; old state immutable.',
        refresh='Full current dense H, deep branches, multi-scale projections and mature readout only on period/age/reset. H/D discarded after refresh. Dense reference runs full same compressed mature model every frame.',
        teacher='Frozen full model on SAME current E/FM during training only; ordinary target discarded, no teacher injection into recurrent Y. Refresh target becomes state by declared policy.',
        loss='Labelled ordinary-frame GT +.2 foreground head KD +.5 feature KD, own two-step TBPTT.',
        optimization='AdamW1e-4 wd1e-4 clip5; plateau factor.5 patience1 threshold.0005 min1e-6; stop min6 epochs and stale4/lr<=2.5e-5; budget is not convergence.',
        validation='Full5794/1738 paired fixed HDY-line baseline, dense full refresh reference, Y-only student. Shared GPU allowed, contaminated timing null. No local skipping or claimed AP until measured.')
    write_json(out/'y_only_contract.json',contract)
    write_json(out/'y_only_engineering.json',engineering(mode,full,cfg,train))
    optimizer=torch.optim.AdamW(updater.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.0005,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    history=[];best=None;significant=-1.;stale=0;start=0;pending=None;full_ap=None
    if a.y_only_resume:
        saved=torch.load(a.y_only_resume,map_location='cpu')
        assert dict(saved['contract'],epochs_max=a.y_only_epochs)==contract
        updater.load_state_dict(saved['updater'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];significant=saved['significant'];stale=saved['stale'];full_ap=saved['full_ap']
        start=saved['epoch'] if saved['next_action']=='evaluate' else saved['epoch']+1
        pending=saved.get('training_summary') if saved['next_action']=='evaluate' else None
        assert start<=a.y_only_epochs
        write_json(out/'resume.json',dict(path=a.y_only_resume,sha256=digest(a.y_only_resume),next_epoch=start,next_action=saved['next_action']))
    for epoch in range(start,a.y_only_epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();losses=[];updates=0
        if pending is not None:losses=pending['losses'];updates=pending['updates'];pending=None
        elif epoch:
            state=None;terms=[];optimizer.zero_grad(set_to_none=True)
            _,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));pred,state,info=mode.step(ego,state,sample['meta'],supervise=True)
                if not info['y_refresh']:
                    with torch.no_grad():teacher=predict(mode.model,info['y_target'])
                    loss=.2*foreground_head_loss(pred,teacher)+.5*feature_distillation(info['y_student'],info['y_target'],teacher)
                    if sample['meta']['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);terms.append(loss);losses.append(float(loss.detach()))
                if len(terms)>=2 or info['y_refresh'] or index+1==len(train):
                    if terms:
                        torch.stack(terms).mean().backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(updater.parameters(),5.))
                        optimizer.step();optimizer.zero_grad(set_to_none=True);terms=[];updates+=1
                    state['y']['value']=state['y']['value'].detach()
                if (index+1)%200==0:
                    evidence=sample_gpu_usage(a.gpu)
                    if not evidence['timing_valid']:write_json(directory/('train_shared_%06d.json'%(index+1)),evidence)
                    write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
            assert index+1==len(train) and losses and not terms
            assert all(p.grad is None for m in (mode.model,mode.readout) for p in m.parameters())
        path=directory/'updater.pth'
        atomic_save(dict(updater=updater.state_dict(),contract=contract,epoch=epoch),path)
        def snapshot(action):return dict(updater=updater.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,contract=contract,best=best,significant=significant,stale=stale,epoch=epoch,next_action=action,full_ap=full_ap)
        atomic_save(dict(snapshot('evaluate'),training_summary=dict(losses=losses,updates=updates)),out/'pending_validation.pth')
        report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,epoch,
            variants=dict(y_full=full,y_only=mode),expected_ap={} if full_ap is None else {'y_full':full_ap},share_gpu=True,
            note='Y-only spatial state plus frozen GRU FM. Ordinary inference bypasses H/D/deblocks/mature readout. Dense small updater, NOT local execution. Hook-calibrated whole-path Conv MACx2. Model timing excludes first10 warmup scans/IO/H2D/postprocess; shared timing null.')
        for name in ('y_full','y_only'):
            result=report['results'][name]
            result['whole_path_full_frames']=result['deep_full_frames'];result['deep_full_frames']=None
        write_json(directory/'report.json',report)
        full_ap=report['results']['y_full']['ap70'];ap=report['results']['y_only']['ap70']
        if epoch:scheduler.step(ap)
        if best is None or ap>best['ap70']:
            best=dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path));write_json(out/'best.json',best)
        if ap>significant+.0005:significant=ap;stale=0
        elif epoch:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,updates=updates,validation=report,lr=optimizer.param_groups[0]['lr'],stale=stale))
        write_json(out/'history.json',history);atomic_save(snapshot('train'),out/'latest.pth')
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=2.5e-5:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=epoch,best=best,stop_reason='plateau' if epoch<a.y_only_epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=epoch))

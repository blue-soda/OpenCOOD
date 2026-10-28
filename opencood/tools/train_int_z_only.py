"""Train signed Z recurrence with an optional trainable mature spatial reader."""
import numpy as np
import torch
from opencood.models.sub_modules.int_z_only import ZOnlyUpdater,ZOnlyDetector,make_reader
from opencood.models.sub_modules.int_anchor_residual import predict
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import evaluate
from opencood.tools.train_int_short_readout import atomic_save
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def load_fixed_reader_start(updater,reader,fixed_reader,initial,source):
    """Keep a joint-trained updater, but explicitly restore the mature reader."""
    original=initial['contract']
    assert original['kind']=='z_only_spatial_v1' and original['reader'] in ('frozen','joint')
    assert original['source']==source and original['width']==64 and original['period']==3
    updater.load_state_dict(initial['updater'],strict=True)
    if original['reader']=='frozen':reader.load_state_dict(initial['reader'],strict=True)
    else:reader.load_state_dict(fixed_reader.state_dict(),strict=True)
    assert all(torch.equal(v,fixed_reader.state_dict()[k]) for k,v in reader.state_dict().items())


def engineering(mode,full,cfg,train):
    state=full_state=native_state=None;samples=[];ordinary=0;maximum=0.;costs={}
    watched=[*(m for m in mode.stage.blocks.modules() if isinstance(m,torch.nn.Conv2d)),
        mode.model.backbone.resnet.layer1,mode.model.backbone.resnet.layer2,
        *mode.model.backbone.deblocks,mode.readout.current,mode.readout.history]
    buffers={k:v.clone() for k,v in mode.reader.named_buffers()}
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            if len(samples)<6:samples.append((ego,meta))
            previous=state;called=[];reads=[]
            hooks=[m.register_forward_hook(lambda m,x,y:called.append(1)) for m in watched]
            hooks.append(mode.reader.register_forward_hook(lambda m,x,y:reads.append(1)))
            counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.updater,mode.reader]))
            try:pred,state,info=mode.step(ego,previous,meta)
            finally:
                counter.close()
                for hook in hooks:hook.remove()
            refresh=info['z_refresh']
            if refresh in costs:assert costs[refresh]==counter.total
            costs[refresh]=counter.total
            assert set(state)=={'fm','z'} and state['z']['value'].shape[1]==96 and len(reads)==1
            if not refresh:ordinary+=1;assert not called,'Ordinary inference executed H/D/projections'
            else:assert called
            other,replayed,_=mode.step(ego,previous,meta,supervise=True)
            assert all(torch.equal(pred[k],other[k]) for k in pred)
            assert torch.equal(state['z']['value'],replayed['z']['value'])
            fc=ConvCounter(full.model,torch.nn.ModuleList([full.readout,full.reader]))
            try:reference,full_state,_=full.step(ego,full_state,meta)
            finally:fc.close()
            assert fc.total==costs[True]
            native,native_state,_=mode.model.step(ego,native_state,meta)
            for k in native:
                maximum=max(maximum,float((native[k]-reference[k]).abs().max()))
                assert torch.allclose(native[k],reference[k],atol=1e-3,rtol=1e-3)
    assert ordinary and set(costs)=={False,True} and costs[False]<costs[True]
    mode.execution_costs=costs;full.execution_costs={True:costs[True]}
    state=None;terms=[]
    for ego,meta in samples:
        pred,state,info=mode.step(ego,state,meta,supervise=True)
        if not info['z_refresh']:terms.append(pred['rm'].square().mean())
    torch.stack(terms).mean().backward()
    for module in (mode.updater,mode.reader):
        params=[p for p in module.parameters() if p.requires_grad]
        if params:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert any(float(p.grad.abs().sum())>0 for p in params)
    assert all(p.grad is None for m in (mode.model,mode.readout,full.reader) for p in m.parameters())
    assert all(torch.equal(buffers[k],v) for k,v in mode.reader.named_buffers())
    mode.updater.zero_grad(set_to_none=True);mode.reader.zero_grad(set_to_none=True)
    return dict(status='PASSED',frames=32,ordinary_frames=ordinary,ordinary_h_d_projection_calls=0,
        reader_every_frame=True,teacher_on_off_exact=True,only_fm_z_persist=True,
        gradients_only_requested_modules=True,reader_bn_statistics_fixed=True,
        full_vs_native_max_error=maximum,full_conv_macs=costs[True],ordinary_conv_macs=costs[False])


def train_z_only(a,cfg,train,val,baseline,out,costs,spatial_macs,source):
    torch.manual_seed(303)
    joint=a.z_only_reader=='joint'
    context=getattr(a,'z_only_context','plain')
    init_path=getattr(a,'z_only_init',None)
    saved_resume=torch.load(a.z_only_resume,map_location='cpu') if a.z_only_resume else None
    warm=None
    if init_path:
        initial=torch.load(init_path,map_location='cpu')
        warm=dict(path=str(init_path),sha256=digest(init_path),epoch=initial['epoch'])
        if initial['contract']['reader']=='joint':
            warm.update(source_reader='joint',reader_initialization='fixed_mature_not_checkpoint')
    elif saved_resume is not None:warm=saved_resume['contract'].get('weights_only_init')
    updater=ZOnlyUpdater(64,context).cuda().eval()
    reader=make_reader(baseline.readout,joint).cuda()
    fixed_reader=make_reader(baseline.readout,False).cuda()
    mode=ZOnlyDetector(baseline,updater,reader,3)
    full=ZOnlyDetector(baseline,updater,fixed_reader,1)
    adapted=ZOnlyDetector(baseline,updater,reader,1)
    contract=dict(kind='z_only_spatial_v1',reader=a.z_only_reader,width=64,period=3,max_age_us=250000,
        seed=303,tbptt=2,epochs_max=a.z_only_epochs,
        source={k:source[k] for k in ('source','memory','readout','repair','basis','data')},
        updater_parameters=sum(p.numel() for p in updater.parameters()),
        reader_parameters=sum(p.numel() for p in reader.parameters()),
        reader_trainable_parameters=sum(p.numel() for p in reader.parameters() if p.requires_grad),
        state='Signed Z96 = frozen shallow projection plus frozen deep projection BEFORE first spatial readout convolution; only FM/Z persist. Own recurrence; ego nearest alignment, period3/.25s.',
        update='E128+oldZ96+valid ->1x1(64)/ReLU ->two3x3(64)/ReLU ->signed1x1(96). g=.95sigmoid*valid; g*old+(1-g)*candidate. Dense5x5 dependencies; no local skipping yet.',
        initialization='Kaiming hidden layers, candidate N(0,.001)/bias0, gate weight0/bias4; mature reader copied exactly. Signed candidate is required by preactivation Z.',
        trainable='Updater plus separate mature spatial reader when joint; frozen reader control differs only in reader trainability. Frozen BN running statistics, trainable reader affine in joint. Pillar/FM/stem/H/deep/projections/heads and independent teacher fixed.',
        teacher='Fixed complete Z from same E/FM; fixed reader and heads produce targets. Ordinary teacher output discarded, never written to student Z. Full refresh writes fixed Z by declared policy.',
        loss='Ordinary: labelled GT +.2 foreground head KD +.5 signedZ feature KD. Joint reader also receives labelled GT +.2 head KD on full refresh to preserve full-path accuracy. No Y feature KD.',
        optimizer='AdamW updater1e-4, joint reader1e-5, wd1e-4 clip5. Plateau factor.5 patience1 absolute.0005 min1e-6. Stop min6/stale4/updater lr<=2.5e-5.',
        validation='Full5794/1738 fixed H control, frozen full Z+reader control, own Z recurrence; joint adds full current Z+adapted reader. Shared GPU timing null. Branch Conv MACsx2 measured; nonconv/Pillar pointwise Linear omitted consistently.')
    if context!='plain' or warm is not None:
        contract.update(context=context,weights_only_init=warm,
            update='Same Z gate and signed candidate; two hidden3x3 layers use dilation1/'+('2' if context=='dilated' else '1')+'; radius'+('3' if context=='dilated' else '2')+'. Equal kernel parameters and nominal MACs; no local execution.',
            initialization='Declared checkpoint weights only; fresh optimizer/scheduler and epoch0. Exact resume retains this original initialization provenance.',
            optimizer='Fresh AdamW updater2.5e-5 wd1e-4 clip5; reader frozen. Same plateau factor.5/patience1/absolute.0005/min1e-6, stop min6/stale4/lr<=2.5e-5.')
        assert not joint and warm is not None
    if init_path:
        load_fixed_reader_start(updater,reader,fixed_reader,initial,contract['source'])
        write_json(out/'initialization.json',dict(warm,context=context,optimizer_reset=True,epoch_reset=True))
    write_json(out/'z_only_contract.json',contract)
    write_json(out/'z_only_engineering.json',engineering(mode,full,cfg,train))
    adapted.execution_costs=full.execution_costs
    groups=[dict(params=list(updater.parameters()),lr=2.5e-5 if warm is not None else 1e-4)]
    if joint:groups.append(dict(params=list(reader.parameters()),lr=1e-5))
    parameters=[p for group in groups for p in group['params']]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.0005,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    history=[];best=None;significant=-1.;stale=0;start=0;pending=None
    if a.z_only_resume:
        saved=saved_resume
        assert dict(saved['contract'],epochs_max=a.z_only_epochs)==contract
        updater.load_state_dict(saved['updater'],strict=True);reader.load_state_dict(saved['reader'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];significant=saved['significant'];stale=saved['stale']
        start=saved['epoch'] if saved['next_action']=='evaluate' else saved['epoch']+1
        pending=saved.get('training_summary') if saved['next_action']=='evaluate' else None
        assert start<=a.z_only_epochs
        write_json(out/'resume.json',dict(path=a.z_only_resume,sha256=digest(a.z_only_resume),next_epoch=start,next_action=saved['next_action']))
    for epoch in range(start,a.z_only_epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir();losses=[];updates=0
        if pending is not None:losses=pending['losses'];updates=pending['updates'];pending=None
        elif epoch:
            state=None;terms=[];optimizer.zero_grad(set_to_none=True)
            _,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'))
                pred,state,info=mode.step(ego,state,sample['meta'],supervise=True)
                if joint or not info['z_refresh']:
                    with torch.no_grad():teacher=predict(mode.model,info['y_target'])
                    loss=.2*foreground_head_loss(pred,teacher)
                    if not info['z_refresh']:loss=loss+.5*feature_distillation(info['z_student'],info['z_target'],teacher)
                    if sample['meta']['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);terms.append(loss);losses.append(float(loss.detach()))
                if len(terms)>=2 or info['z_refresh'] or index+1==len(train):
                    if terms:
                        torch.stack(terms).mean().backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,5.))
                        optimizer.step();optimizer.zero_grad(set_to_none=True);terms=[];updates+=1
                    state['z']['value']=state['z']['value'].detach()
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
            assert index+1==len(train) and losses and not terms
            assert all(p.grad is None for m in (mode.model,mode.readout,fixed_reader) for p in m.parameters())
        path=directory/'updater.pth'
        atomic_save(dict(updater=updater.state_dict(),reader=reader.state_dict(),contract=contract,epoch=epoch),path)
        def snapshot(action):return dict(updater=updater.state_dict(),reader=reader.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),history=history,contract=contract,best=best,significant=significant,stale=stale,epoch=epoch,next_action=action)
        atomic_save(dict(snapshot('evaluate'),training_summary=dict(losses=losses,updates=updates)),out/'pending_validation.pth')
        variants=dict(z_full_fixed=full,z_only=mode)
        if joint:variants['z_full_adapted']=adapted
        report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,epoch,variants=variants,
            expected_ap={'z_full_fixed':.7567910702899006},share_gpu=True,
            note='Signed Z before mature spatial reader; ordinary H/D/projections skipped, reader dense every frame. Independent teacher and histories. Joint readout has fixed BN statistics. No local execution yet; Conv MACsx2; shared timing null.')
        for name in variants:
            result=report['results'][name];result['z_full_frames']=result['deep_full_frames'];result['deep_full_frames']=None
            result['reader_dense_frames']=result['readout_full_frames'];result['readout_full_frames']=None
        write_json(directory/'report.json',report)
        ap=report['results']['z_only']['ap70']
        if epoch:scheduler.step(ap)
        if best is None or ap>best['ap70']:
            best=dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path));write_json(out/'best.json',best)
        if ap>significant+.0005:significant=ap;stale=0
        elif epoch:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,updates=updates,validation=report,
            lrs=[group['lr'] for group in optimizer.param_groups],stale=stale))
        write_json(out/'history.json',history);atomic_save(snapshot('train'),out/'latest.pth')
        stopped=epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=2.5e-5
        if stopped:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=epoch,best=best,stop_reason='plateau' if stopped else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=epoch))

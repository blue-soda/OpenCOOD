"""Train only small residual repair modules; original ANN GRU stays frozen."""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_light_repair import make_repairs,repair_input_request
from opencood.models.sub_modules.int_local_cache import LocalCacheDetector,local_change_mask,OriginalDetectorControl
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_cache_repair import tile_mask,expand_tiles
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.run_int_cache_repair import evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def evaluate_saved(a,model,cfg,rows,out,policy,provenance):
    """Post-training full evaluation; independently reload immutable repair weights."""
    checkpoint=Path(a.repair_checkpoint)
    run=checkpoint.parent.parent
    assert json.loads((run/'status.json').read_text())['status']=='COMPLETED', 'Finish training before paired benchmarking'
    original=json.loads((run/'provenance.json').read_text())
    assert original['contract']==provenance['contract']
    assert original['checkpoint']['sha256']==provenance['checkpoint']['sha256']
    saved=torch.load(str(checkpoint),map_location='cpu');contract=saved['contract']
    assert contract['source_policy']==policy
    assert contract['teacher_checkpoint_sha256']==provenance['checkpoint']['sha256']
    assert contract['repair_policy']==a.repair_policy
    repairs=make_repairs().cuda().eval();repairs.load_state_dict(saved['repairs'],strict=True)
    assert sum(p.numel() for p in repairs.parameters())==contract['trainable_parameters']
    ablation=getattr(a,'repair_ablation','none')
    if ablation=='zero-residual':
        assert a.repair_period==2 and a.repair_policy=='dense'
        # Preserve all input preparation and small-module compute, but suppress
        # learned correction. Outputs are exactly the warped historical values.
        with torch.no_grad():
            for repair in repairs:
                repair.net[-1].weight.zero_();repair.net[-1].bias.zero_()
    if ablation=='original':assert a.repair_period==1
    engineering_report=json.loads((run/'engineering.json').read_text())
    assert engineering_report['status']=='PASSED'
    if a.repair_period==3:
        assert a.repair_policy=='dense' and ablation=='none'
        engineering(model,cfg,rows,repairs,policy['selected']['threshold'],out,True,period=3)
    profiling=getattr(a,'profile_stages',False)
    a.stage='pilot' if profiling else 'full';a.engineering=str(run/'engineering.json')
    wrapper=LocalCacheDetector(model,cfg,policy['selected']['threshold'],period=a.repair_period,
                              repair_modules=repairs,repair_dense=a.repair_policy=='dense')
    if ablation=='original':wrapper=OriginalDetectorControl(model)
    torch.cuda.reset_peak_memory_stats()
    if profiling:
        from opencood.tools.int_repair_stage_profile import RepairStageProfiler
        profiler=RepairStageProfiler(wrapper,model,repairs)
        try:
            report=evaluate(a,model,cfg,rows,out,profiler)
        finally:
            profiler.close()
        report['stage_profile']=profiler.summary()
    else:
        report=evaluate(a,model,cfg,rows,out,wrapper)
    frames=[json.loads(s) for s in (out/'frames.jsonl').read_text().splitlines()]
    partial=[r for r in frames if r['local_partial']]
    report.update(repair_policy=a.repair_policy,repair_period=a.repair_period,repair_checkpoint=str(checkpoint),
        repair_ablation=ablation,
        repair_checkpoint_sha256=digest(checkpoint),repair_epoch=saved['epoch'],
        partial_frames=len(partial),partial_labeled_frames=sum(r['supervised'] for r in partial),
        original_deep_stage_call_fraction=1-len(partial)/len(frames),
        caveats=['Reloaded frozen detector and repair checkpoint; period1 is paired all-original-stage control.',
                 'Period2 skips original deep stages on every partial frame; local=actual tile crops, dense=whole small repair modules.',
                 'stage_means combines different module input areas, not backbone MAC reduction.',
                 'Timing includes model/gate/warp/cache/repair/decoder/head but excludes IO/voxelization/postprocess.',
                 'Use sequential runs on the same GPU after both training jobs finish; one timing pair is not a stability study.'])
    if ablation=='zero-residual':
        report['caveats'].append('Loaded checkpoint hash is unchanged provenance; final residual weights/bias are zeroed in memory. Small-module compute remains, so this is an accuracy intervention, not an optimized warp-only speed baseline. New-view deep features remain zero.')
    if ablation=='original':
        report['caveats'].append('Direct original model.step; no deep cache or repair execution. Only existing FM state is retained. Loaded repair modules are unused.')
    if a.repair_period==3:
        training_period=contract.get('repair_period',2)
        report['caveats'].append('Period3 evaluation with period%d-trained weights; %s. Up to two partial frames use the same full anchor, age below 0.25s; repaired outputs are never written into the anchor.' % (training_period,'matched interval' if training_period==3 else 'zero-shot interval extrapolation'))
    if profiling:report['caveats'].extend(report['stage_profile']['caveats'])
    return report


def training_pair(first,targets,teacher,cache,meta,repairs,bounds,threshold,dense=False):
    if dense:
        changed=torch.zeros_like(first[:,:1],dtype=torch.bool)
    else:
        desc=early_descriptor(first,4)
        aligned,valid=warp_bev(cache['desc'],cache['pose'],meta['pose'],bounds)
        changed=expand_tiles(local_change_mask(desc,aligned,valid,threshold),4,first.shape[-2:])
    outputs=[];loss=first.new_zeros(())
    for index,repair in enumerate(repairs,1):
        old,valid=warp_bev(cache['values'][index-1],cache['pose'],meta['pose'],bounds)
        inputs,request=repair_input_request(first,old,valid,changed,index)
        written=expand_tiles(tile_mask(request,4),4,old.shape[-2:])
        # Dense training is allowed; measured inference uses actual crop execution.
        prediction=repair(inputs) if dense else torch.where(written,repair(inputs),old)
        target=targets[index-1]
        weight=1+4*F.adaptive_avg_pool2d(teacher['psm'].sigmoid().amax(1,keepdim=True),target.shape[-2:])
        loss=loss+((prediction-target).square()*weight).mean()/target.square().mean().clamp_min(1e-4)
        outputs.append(prediction)
    return outputs,loss


def engineering(model,cfg,rows,repairs,threshold,out,dense=False,period=2):
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(test.stdout)
    if test.returncode:raise RuntimeError('INT tests failed')
    full=LocalCacheDetector(model,cfg,threshold,period=1,repair_modules=repairs,repair_dense=dense)
    partial=LocalCacheDetector(model,cfg,threshold,period=period,verify=True,repair_modules=repairs,repair_dense=dense)
    states=[None,None,None];errors=[];crop_errors=[];deep_calls=[0,0];opportunities=0;max_partial_depth=0
    def hit(i):
        def callback(module,inputs,output):deep_calls[i]+=1
        return callback
    hooks=[getattr(model.backbone.resnet,'layer%d'%(i+1)).register_forward_hook(hit(i)) for i in range(2)]
    _,stream=loader(rows[:8],cfg,2,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                expected,states[0],_=model.step(ego,states[0],meta)
                actual,states[1],_=full.step(ego,states[1],meta)
                error=max(float((actual[k]-expected[k]).abs().max()) for k in actual)
                assert error<=1e-5;errors.append(error)
                before=list(deep_calls)
                previous=None if states[2] is None else states[2]['cache']
                actual,states[2],info=partial.step(ego,states[2],meta)
                calls=[a-b for a,b in zip(deep_calls,before)]
                assert calls==([0,0] if info['local_partial'] else [1,1])
                if info['local_partial']:
                    assert states[2]['cache']['stamp']==previous['stamp']
                    assert all(x is y for x,y in zip(states[2]['cache']['values'],previous['values']))
                    assert 0 < info['cache_age_seconds'] < .25
                    assert info['cache_frames_since_refresh'] < period
                    max_partial_depth=max(max_partial_depth,info['cache_frames_since_refresh'])
                assert all(torch.isfinite(v).all() for v in actual.values())
                opportunities+=int(info['local_partial']);crop_errors.extend(info['local_verification_errors'])
    finally:
        for hook in hooks:hook.remove()
    if period==3:assert max_partial_depth==2, 'Engineering prefix must exercise two repairs from one anchor'
    report=dict(status='PASSED',period=period,max_partial_depth=max_partial_depth,full_refresh_errors=errors,crop_errors=crop_errors,
                partial_frames_with_zero_original_deep_stage_calls=opportunities)
    write_json(out/'engineering.json',report)


def train(a,model,cfg,train_rows,val_rows,out,policy,provenance):
    threshold=policy['selected']['threshold'];bounds=cfg['model']['args']['lidar_range']
    torch.manual_seed(303);repairs=make_repairs().cuda().eval()
    period=a.repair_period
    assert period in (2,3) and (period==2 or a.repair_policy=='dense')
    initialization=None
    if a.repair_checkpoint:
        initial_path=Path(a.repair_checkpoint);initial_run=initial_path.parent.parent
        assert json.loads((initial_run/'status.json').read_text())['status']=='COMPLETED'
        initial_best=json.loads((initial_run/'best.json').read_text())
        assert initial_best['checkpoint']==str(initial_path) and initial_best['sha256']==digest(initial_path)
        initial_provenance=json.loads((initial_run/'provenance.json').read_text())
        assert initial_provenance['contract']==provenance['contract']
        initial=torch.load(str(initial_path),map_location='cpu')
        assert initial['contract']['teacher_checkpoint_sha256']==provenance['checkpoint']['sha256']
        assert initial['contract']['source_policy']==policy and initial['contract']['repair_policy']==a.repair_policy
        repairs.load_state_dict(initial['repairs'],strict=True)
        initialization=dict(checkpoint=str(initial_path),sha256=initial_best['sha256'],
                            source_period=initial['contract'].get('repair_period',2),source_epoch=initial['epoch'],
                            optimizer='Fresh optimizer/scheduler; not an interrupted-run resume')
    for parameter in model.parameters():parameter.requires_grad_(False)
    dense=a.repair_policy=='dense'
    engineering(model,cfg,val_rows,repairs,threshold,out,dense,period=period)
    optimizer=torch.optim.AdamW(repairs.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,
                                                       threshold=.001,threshold_mode='abs',min_lr=1e-5)
    contract=dict(source_policy=policy,repair_policy=a.repair_policy,repair_period=period,initialization=initialization,
        seed=303,epochs_min=5,epochs_max=20,patience=5,min_delta=.001,
        optimizer='AdamW',initial_lr=.001,weight_decay=.0001,
        teacher_checkpoint_sha256=provenance['checkpoint']['sha256'],
        train_scans=len(train_rows),val_scans=len(val_rows),trainable_parameters=sum(p.numel() for p in repairs.parameters()),
        frozen_parameters=sum(p.numel() for p in model.parameters()),
        loss='Normalized feature MSE weighted by frozen teacher objectness + 0.1 weighted cls/reg output smooth L1',
        protocol='Full-frame anchors alternate with at most %d repair frames, age below 0.25s; original model always eval/frozen; no augmentation or artificial delays' % (period-1),
        inference='Independent stage2/3 residual repair; local=tile4 radius1 crops, dense=whole small module without gate; no original deep stages or teacher on repair frames')
    write_json(out/'training_contract.json',contract)
    best=-1.;significant=-1.;stale=0;history=[];captured={}
    def capture(i):
        def callback(module,inputs,output):captured[i]=output.detach()
        return callback
    hooks=[getattr(model.backbone.resnet,'layer%d'%i).register_forward_hook(capture(i)) for i in range(3)]
    try:
        for epoch in range(1,21):
            repairs.train();state=cache=None;losses=[];started=time.monotonic();updates=0
            losses_by_depth={i:[] for i in range(1,period)}
            _,stream=loader(train_rows,cfg,4,303+epoch)
            for index,(row,sample) in enumerate(zip(train_rows,stream)):
                meta=sample['meta'];assert row['frame']==meta['frame']
                ego=to_device(sample['ego'],torch.device('cuda'))
                with torch.no_grad():
                    teacher,state,info=model.step(ego,state,meta)
                    first=captured[0];targets=[captured[1],captured[2]]
                    full=cache is None or bool(info['reset']) or cache['frames']>=period-1 or (int(meta['timestamp_us'])-cache['stamp'])/1e6>=.25
                    if full:
                        cache=dict(values=targets,desc=early_descriptor(first,4),pose=meta['pose'],
                                   stamp=int(meta['timestamp_us']),frames=0)
                if not full:
                    optimizer.zero_grad(set_to_none=True)
                    outputs,loss=training_pair(first,targets,teacher,cache,meta,repairs,bounds,threshold,dense)
                    features=model.backbone.decode_multiscale_feature([first]+outputs)
                    if model.shrink_flag:features=model.shrink_conv(features)
                    predictions=dict(psm=model.cls_head(features),rm=model.reg_head(features))
                    weight=1+4*teacher['psm'].sigmoid().amax(1,keepdim=True)
                    for key in predictions:
                        loss=loss+.1*(F.smooth_l1_loss(predictions[key],teacher[key],reduction='none')*weight).mean()
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Non-finite repair loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(repairs.parameters(),5.)
                    if not bool(torch.isfinite(norm)):raise RuntimeError('Non-finite repair gradients')
                    optimizer.step();losses.append(float(loss.detach()));updates+=1;cache['frames']+=1
                    losses_by_depth[cache['frames']].append(losses[-1])
                if (index+1)%200==0:
                    write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,
                        total_scans=len(train_rows),updates=updates,updates_by_depth={k:len(v) for k,v in losses_by_depth.items()},
                        mean_loss=float(np.mean(losses)) if losses else None))
                    print('TRAIN',epoch,index+1,'loss',float(np.mean(losses)) if losses else None,flush=True)
            assert index+1==len(train_rows) and updates>0
            assert all(losses_by_depth.values()), 'Every configured repair depth must be exercised'
            assert all(p.grad is None for p in model.parameters())
            epoch_dir=out/('epoch_%03d'%epoch);epoch_dir.mkdir()
            repairs.eval();checkpoint=epoch_dir/'repairs.pth'
            torch.save(dict(repairs=repairs.state_dict(),epoch=epoch,contract=contract),str(checkpoint))
            write_json(out/'status.json',dict(status='RUNNING',phase='full_validation',epoch=epoch))
            a.stage='full';a.engineering=str(out/'engineering.json')
            wrapper=LocalCacheDetector(model,cfg,threshold,period=period,repair_modules=repairs,repair_dense=dense)
            torch.cuda.reset_peak_memory_stats()
            report=evaluate(a,model,cfg,val_rows,epoch_dir,wrapper)
            records=[json.loads(s) for s in (epoch_dir/'frames.jsonl').read_text().splitlines()]
            partial=[r for r in records if r['local_partial']]
            report.update(repair_checkpoint_sha256=digest(checkpoint),epoch=epoch,repair_policy=a.repair_policy,repair_period=period,
                partial_frames=len(partial),partial_labeled_frames=sum(r['supervised'] for r in partial),
                original_deep_stage_call_fraction=1-len(partial)/len(records),
                caveats=['Learned ANN repair, not independent SNN benefit. Only repair modules trained; source detector frozen.',
                         'Original stages2/3 never called on repair frames; local uses actual radius1 tiles, dense uses the complete small repair modules.',
                         'stage_means mixes full-backbone input ratios on full frames with repair-module ratios on partial frames; not backbone MAC.',
                         'Model timing includes warp/selection/repair/cache/decoder/head, excludes input and postprocess; single measurement.'])
            write_json(epoch_dir/'report.json',report)
            ap=report['metrics']['bev_ap70'];scheduler.step(ap)
            if ap>best:
                best=ap;write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(checkpoint),sha256=digest(checkpoint)))
            if ap>significant+.001:significant=ap;stale=0
            else:stale+=1
            record=dict(epoch=epoch,train_loss=float(np.mean(losses)),updates=updates,ap70=ap,
                updates_by_depth={k:len(v) for k,v in losses_by_depth.items()},
                train_loss_by_depth={k:float(np.mean(v)) for k,v in losses_by_depth.items()},
                lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started,stale_epochs=stale)
            history.append(record);write_json(out/'history.json',history)
            torch.save(dict(repairs=repairs.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                epoch=epoch,best=best,significant=significant,stale=stale,history=history,contract=contract),str(out/'latest.pth'))
            if epoch>=5 and stale>=5 and optimizer.param_groups[0]['lr']<=.0001:break
    finally:
        for hook in hooks:hook.remove()
    return dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),
                stop_reason='plateau' if stale>=5 and len(history)>=5 and optimizer.param_groups[0]['lr']<=.0001 else 'max_epochs_not_proof_of_convergence')

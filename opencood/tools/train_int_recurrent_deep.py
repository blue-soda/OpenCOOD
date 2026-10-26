"""Train short complete-D recurrence while preserving the mature H baseline."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_gpu_exclusivity import check_gpu_exclusive,wait_for_training_gpu
from opencood.utils import eval_utils


def cost_of(costs,spatial_macs,info,recurrent):
    partial=info['deep_partial'] if recurrent else info['local_partial']
    value=costs[('dense',partial)]
    if not info['spatial_refresh']:value=value-spatial_macs+info['spatial_extra_linear_macs']
    return value-info.get('deep_macs_saved',0)-info.get('readout_macs_saved',0)


def engineering(baseline,mode,cfg,train,costs,spatial_macs):
    one=RecurrentDeepDetector(baseline,mode.repair,1);control=copy.copy(baseline);control.deep_period=1
    states={k:None for k in ('one','control','recurrent','supervised')};maximum=0.;partial=0;samples=[]
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            if len(samples)<8:samples.append((ego,meta))
            counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
            try:pred,states['recurrent'],info=mode.step(ego,states['recurrent'],meta)
            finally:counter.close()
            assert counter.total+info['spatial_extra_linear_macs']==cost_of(costs,spatial_macs,info,True)
            other,states['supervised'],_=mode.step(ego,states['supervised'],meta,supervise=True)
            for k in pred:
                maximum=max(maximum,float((pred[k]-other[k]).abs().max()));assert torch.equal(pred[k],other[k])
            assert torch.equal(states['recurrent']['deep']['value'],states['supervised']['deep']['value'])
            left,states['one'],_=one.step(ego,states['one'],meta)
            right,states['control'],_=control.step(ego,states['control'],meta)
            for k in left:assert torch.equal(left[k],right[k]),'period1 changed mature model'
            partial+=int(info['deep_partial'])
    mode.repair.zero_grad(set_to_none=True);state=None;losses=[]
    for ego,meta in samples:
        pred,state,info=mode.step(ego,state,meta,supervise=True)
        if info['deep_partial']:losses.append(pred['rm'].square().mean())
    assert losses and partial
    torch.stack(losses).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mode.repair.parameters())
    assert float(mode.repair.net[-1].weight.grad.abs().sum())>0
    assert all(p.grad is None for m in (mode.model,mode.readout) for p in m.parameters())
    mode.repair.zero_grad(set_to_none=True)
    return dict(status='PASSED',frames=32,partial_frames=partial,period1_exact=True,
        train_inference_exact_max_error=maximum,executed_macs_verified=True,
        gradient_into_repair_only=True,teacher_not_used_in_partial_inference=True)


def evaluate(a,baseline,mode,cfg,rows,directory,costs,spatial_macs,epoch,variants=None,expected_ap=None,note=None,audit=None,wait_for_gpu=False):
    modes=dict(baseline=baseline);modes.update(variants if variants is not None else dict(recurrent=mode));states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    metrics={k:dict(time=[],macs=0,full=0,state=[],deep_write=[],short_ratio=[],readout_write=[],readout_ratio=[],readout_full=0) for k in modes};isolation=[]
    dataset,stream=loader(rows,cfg,4,303)
    def guard(scan):
        try:
            if wait_for_gpu:
                # Pause only between complete frames. All trajectory states and
                # AP accumulators stay intact; any wait invalidates whole-run time.
                def waiting(event):
                    write_json(directory/('validation_wait_%06d.json'%scan),event)
                    write_json(directory.parent/'status.json',dict(status='RUNNING',
                        phase='waiting_validation_gpu',epoch=epoch,scans=scan))
                evidence=wait_for_training_gpu(a.gpu,on_event=waiting)
                write_json(directory.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=scan))
            else:evidence=check_gpu_exclusive(a.gpu)
            evidence['scan']=scan;isolation.append(evidence)
            write_json(directory/'gpu_exclusivity.json',dict(status='SAMPLED_CLEAR' if all(x['timing_valid'] for x in isolation) else 'INVALID_TIMING',samples=isolation))
        except Exception as exc:
            write_json(directory/'gpu_exclusivity.json',dict(status='INVALID_TIMING',scan=scan,error=repr(exc),samples=isolation,evidence=getattr(exc,'evidence',[])))
            raise
    guard(0)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));names=list(modes)
            if variants is not None:
                offset=index%len(names);names=names[offset:]+names[:offset]
                if (index//len(names))%2:names.reverse()
            elif index%2:names.reverse()
            for name in names:
                previous=states[name]
                torch.cuda.synchronize();tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,previous,sample['meta'])
                torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                if index>=10:metrics[name]['time'].append(ms)
                recurrent=name!='baseline';metrics[name]['macs']+=cost_of(costs,spatial_macs,info,recurrent)
                metrics[name]['full']+=int(not info['deep_partial'] if recurrent else not info['local_partial'])
                metrics[name]['state'].append(info['state_bytes']+info['spatial_state_bytes']+info['anchor_bytes']+info.get('readout_state_bytes',0))
                if 'readout_write_fraction' in info:
                    metrics[name]['readout_write'].append(info['readout_write_fraction'])
                    metrics[name]['readout_ratio'].append(info['readout_executed_fraction'])
                    metrics[name]['readout_full']+=int(info['readout_refresh'])
                if 'deep_write_fraction' in info:metrics[name]['deep_write'].append(info['deep_write_fraction'])
                if 'deep_short_macs_ratio' in info:metrics[name]['short_ratio'].append(info['deep_short_macs_ratio'])
                if audit is not None:audit(name,ego,previous,sample['meta'],pred,states[name],info,index)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if sample['meta']['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if (index+1)%200==0:
                guard(index+1);write_json(directory.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=index+1))
    assert index+1==len(rows);guard(index+1)
    timing_valid=all(x['timing_valid'] for x in isolation)
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        conv_linear_gflops=2*metrics[k]['macs']/len(rows)/1e9,mean_ms=float(np.mean(metrics[k]['time'])) if timing_valid else None,
        p95_ms=float(np.percentile(metrics[k]['time'],95)) if timing_valid else None,deep_full_frames=metrics[k]['full'],
        mean_state_bytes=float(np.mean(metrics[k]['state'])),
        mean_readout_write_fraction=float(np.mean(metrics[k]['readout_write'])) if metrics[k]['readout_write'] else None,
        mean_readout_executed_fraction=float(np.mean(metrics[k]['readout_ratio'])) if metrics[k]['readout_ratio'] else None,
        readout_full_frames=metrics[k]['readout_full'] if metrics[k]['readout_write'] else None,
        readout_packed_weight_bytes=modes[k].readout_cache.packed_weight_bytes if getattr(modes[k],'readout_cache',None) else 0,
        readout_geometry_bytes=getattr(getattr(modes[k],'readout_cache',None),'geometry_bytes',0),
        mean_deep_write_fraction=float(np.mean(metrics[k]['deep_write'])) if metrics[k]['deep_write'] else None,
        mean_short_update_macs_ratio=float(np.mean(metrics[k]['short_ratio'])) if metrics[k]['short_ratio'] else None) for k in modes}
    assert abs(results['baseline']['ap70']-.7553159775482435)<1e-7
    for name,ap in (expected_ap or {}).items():
        assert abs(results[name]['ap70']-ap)<1e-7,(name,results[name]['ap70'],ap)
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,timing_valid=timing_valid,
        note=note or 'Independent histories. Dense short D update; no local D skipping. Timing excludes IO/H2D/postprocess; sampled GPU isolation does not reserve shared CPU/IO. Conv/functional-linear MACsx2.')
    write_json(directory/'report.json',report);return report


def train_recurrent(a,cfg,train,val,baseline,out,costs,spatial_macs):
    torch.manual_seed(303);repair=copy.deepcopy(baseline.repair).eval()
    for p in repair.parameters():p.requires_grad_(True)
    mode=RecurrentDeepDetector(baseline,repair,a.recurrent_deep_period)
    provenance=json.loads((out/'contract.json').read_text())
    contract=dict(source={k:provenance[k] for k in ('source','memory','readout','repair','basis','data')},
        seed=303,period=a.recurrent_deep_period,max_full_age_us=mode.deep.max_age_us,
        trainable_parameters=sum(p.numel() for p in repair.parameters()),tbptt=3,epochs_max=a.recurrent_deep_epochs,
        state='Complete D written back every frame. Current H/spatial selection/own confidence and FM remain frozen mature architecture; each trajectory owns all states.',
        teacher='Full deep projection and head on student current H only for discarded training targets; enters student state only on declared full refresh. No teacher-forced ordinary state.',
        loss='GT on labelled partial frames + .2 head KD + .5 feature KD. Mean loss over up to3 actual partial updates per optimizer step.',
        optimization='Pretrained width32 repair; AdamW lr1e-4 wd1e-4 clip5. Plateau factor.5 patience1 min1e-6 threshold.0005; min6 epochs and stale4/lr<=2.5e-5. Maximum budget is not convergence.',
        inference='Dense short updater first; no readout cache, no local D updater yet. Preserve fixed four-layer H baseline and period3 immutable-anchor paired control.')
    write_json(out/'recurrent_training_contract.json',contract)
    gate=engineering(baseline,mode,cfg,train,costs,spatial_macs);write_json(out/'recurrent_engineering.json',gate)
    optimizer=torch.optim.AdamW(repair.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1,threshold=.0005,threshold_mode='abs',min_lr=1e-6)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();history=[];best_ap=significant=-1.;stale=0;best=None
    start=0;pending_evaluation=None
    if a.recurrent_deep_resume:
        saved=torch.load(a.recurrent_deep_resume,map_location='cpu')
        assert dict(saved['contract'],epochs_max=a.recurrent_deep_epochs)==contract
        repair.load_state_dict(saved['repair'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];best=saved['best'];best_ap=best['ap70'] if best is not None else -1.;significant=saved['significant'];stale=saved['stale']
        if saved.get('next_action')=='evaluate':
            start=saved['epoch'];pending_evaluation=saved['training_summary']
        else:start=history[-1]['epoch']+1
        if start>a.recurrent_deep_epochs:raise ValueError('Resume requires a larger epoch budget')
        if best is not None:write_json(out/'best.json',best)
        write_json(out/'resume.json',dict(checkpoint=str(Path(a.recurrent_deep_resume).resolve()),sha256=digest(a.recurrent_deep_resume),
            next_epoch=start,next_action=saved.get('next_action','train'),scope='Optimizer/scheduler/model continuation; pending validation resumes without repeating training; interrupted partial training epoch restarts.'))
    for epoch in range(start,a.recurrent_deep_epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir()
        losses=[];depths={};updates=0
        if pending_evaluation is not None and epoch==start:
            losses=pending_evaluation['losses'];depths=pending_evaluation['depths'];updates=pending_evaluation['updates']
        elif epoch:
            state=None;pending=[];optimizer.zero_grad(set_to_none=True)
            _,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'))
                pred,state,info=mode.step(ego,state,sample['meta'],supervise=True)
                if info['deep_partial']:
                    loss=.2*foreground_head_loss(pred,info['target_predictions'])+.5*feature_distillation(info['student_deep'],info['target_deep'],info['target_predictions'])
                    if sample['meta']['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                    assert torch.isfinite(loss);pending.append(loss);losses.append(float(loss.detach()))
                    depth=info['deep_depth'];depths[depth]=depths.get(depth,0)+1
                boundary=len(pending)>=3 or not info['deep_partial'] or index+1==len(train)
                if boundary:
                    if pending:
                        torch.stack(pending).mean().backward()
                        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(repair.parameters(),5.))
                        optimizer.step();optimizer.zero_grad(set_to_none=True);pending=[];updates+=1
                    state['deep']['value']=state['deep']['value'].detach()
                if (index+1)%200==0:
                    torch.cuda.synchronize()
                    def waiting(event):
                        write_json(directory/('train_wait_%06d.json'%(index+1)),event)
                        write_json(out/'status.json',dict(status='RUNNING',phase='waiting_gpu' if event['status']=='WAITING_GPU' else event['status'].lower(),epoch=epoch,scans=index+1))
                    try:
                        evidence=wait_for_training_gpu(a.gpu,on_event=waiting)
                        if not evidence['timing_valid']:
                            write_json(directory/('train_contention_%06d.json'%(index+1)),evidence)
                    except Exception as exc:
                        write_json(directory/'train_contention.json',dict(scan=index+1,error=repr(exc),evidence=getattr(exc,'evidence',[])))
                        raise
                    write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses))))
            assert index+1==len(train) and losses and not pending
            assert all(p.grad is None for m in (mode.model,mode.readout) for p in m.parameters())
        path=directory/'repair.pth';torch.save(dict(repair=repair.state_dict(),contract=contract,epoch=epoch),str(path))
        torch.save(dict(repair=repair.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,contract=contract,best=best,significant=significant,stale=stale,epoch=epoch,next_action='evaluate',
            training_summary=dict(losses=losses,depths=depths,updates=updates)),str(out/'pending_validation.tmp'))
        (out/'pending_validation.tmp').replace(out/'pending_validation.pth')
        report=evaluate(a,baseline,mode,cfg,val,directory,costs,spatial_macs,epoch)
        ap=report['results']['recurrent']['ap70']
        if epoch:scheduler.step(ap)
        if ap>best_ap:
            best_ap=ap;best=dict(ap70=ap,epoch=epoch,checkpoint=str(path),sha256=digest(path));write_json(out/'best.json',best)
        if ap>significant+.0005:significant=ap;stale=0
        elif epoch:stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,updates=updates,depths=depths,
            validation=report,lr=optimizer.param_groups[0]['lr'],stale=stale))
        write_json(out/'history.json',history)
        torch.save(dict(repair=repair.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,contract=contract,best=best,significant=significant,stale=stale,next_action='train'),str(out/'latest.pth'))
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=2.5e-5:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=epoch,best=best,
        stop_reason='plateau' if epoch<a.recurrent_deep_epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=epoch))

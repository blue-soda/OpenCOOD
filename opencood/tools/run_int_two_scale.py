"""Train a direct-readout recurrent BEV baseline, before adding selective updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_two_scale import TwoScaleDetector
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector, GatedPyramidStateDetector
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path, relocate_tree
from opencood.tools.run_int_local_state import save_checkpoint
import signal
import random


def make_optimizer(student, profile='uniform'):
    if profile == 'uniform':
        return torch.optim.AdamW(student.core.parameters(),lr=.0003,weight_decay=.0001)
    if profile not in ('transfer','protect_spatial'): raise ValueError(profile)
    pyramid=isinstance(student,PyramidStateDetector)
    if profile == 'protect_spatial' and not pyramid:
        raise ValueError('protect_spatial is registered only for pyramid models')
    names=('new','observation','coarse_encoder','heads') + (('decoder',) if pyramid else ())
    buckets={name:[] for name in names}
    for name,param in student.core.named_parameters():
        if not param.requires_grad: continue
        prefix=name.split('.')[0]
        group=(prefix if prefix in ('observation','coarse_encoder') else
               'heads' if prefix in ('cls_head','reg_head','dir_head') else 'new')
        if pyramid and prefix=='deep_encoder':group='coarse_encoder'
        if pyramid and prefix in ('deblocks','readout'):group='decoder'
        buckets[group].append(param)
    rates=dict(new=.0003,observation=.00001,coarse_encoder=.0001,heads=.00001)
    if pyramid:rates['decoder']=.00001
    if profile == 'protect_spatial':rates['coarse_encoder']=.00001
    groups=[dict(params=params,lr=rates[name],name=name) for name,params in buckets.items()]
    expected={id(p) for p in student.core.parameters() if p.requires_grad}
    actual=[id(p) for group in groups for p in group['params']]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    assert all(group['params'] for group in groups)
    return torch.optim.AdamW(groups,weight_decay=.0001)


def detach_stream_state(state):
    return {key:value.detach() if torch.is_tensor(value) else value for key,value in state.items()}


def engineering(student,teacher,cfg,rows,out,optimizer_profile='uniform'):
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:raise RuntimeError('INT engineering tests failed')
    torch.cuda.reset_peak_memory_stats()
    calls=[0,0,0]
    def hit(index):
        def callback(module,inputs,output):calls[index]+=1
        return callback
    modules=[teacher.backbone.resnet,teacher.backbone.deblocks[0],teacher.shrink_conv]
    hooks=[m.register_forward_hook(hit(i)) for i,m in enumerate(modules)]
    student.eval();state=None;records=[];conv_macs={};cost_hooks=[];capacity_parity=[];spatial_parity=[]
    pyramid=isinstance(student,PyramidStateDetector)
    initial_pyramid=pyramid and student.updates_are_initial_identity()
    def count_conv(name):
        def callback(module,inputs,output):
            group=name.split('.')[0]
            macs=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            if isinstance(module,torch.nn.ConvTranspose2d):
                macs=inputs[0].numel()*(module.out_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            conv_macs[group]=conv_macs.get(group,0)+macs
        return callback
    for name,module in student.core.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):cost_hooks.append(module.register_forward_hook(count_conv(name)))
    _,stream=loader(rows[:8],cfg,2,303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'))
                prior_state=state
                pred,state,info=student.step(ego,state,sample['meta'])
                for hook in cost_hooks:hook.remove()
                cost_hooks=[]
                if initial_pyramid:
                    student.spatial_only=True
                    try:control,control_state,_=student.step(ego,prior_state,sample['meta'])
                    finally:student.spatial_only=False
                    assert all(torch.equal(pred[k],control[k]) for k in pred)
                    assert all(torch.equal(state[k],control_state[k]) for k in student.state_keys)
                    spatial_parity.append(sample['meta']['frame'])
                if student.variant == 'two_scale_medium_context_v1':
                    context=student.core.coarse_context
                    # An initial zero branch must be functionally identical to the small core.
                    # On checkpoint evaluation the learned branch is intentionally nonzero.
                    if not torch.count_nonzero(context.projection.weight) and not torch.count_nonzero(context.projection.bias):
                        student.core.coarse_context=torch.nn.Identity()
                        try:control,control_state,_=student.step(ego,prior_state,sample['meta'])
                        finally:student.core.coarse_context=context
                        assert all(torch.equal(pred[k],control[k]) for k in pred)
                        assert all(torch.equal(state[k],control_state[k]) for k in ('value','coarse'))
                        capacity_parity.append(sample['meta']['frame'])
                assert all(torch.isfinite(v).all() for v in pred.values())
                assert pred['psm'].shape[-2:]==ego['anchor_box'].shape[:2]
                assert calls==[0,0,0]
                records.append(dict(frame=sample['meta']['frame'],reset=info['reset'],
                                    state_shape=list(state['value'].shape)))
    finally:
        for hook in hooks:hook.remove()
        for hook in cost_hooks:hook.remove()
    assert not any('backbone' in name or 'feature_memory' in name for name,_ in student.named_modules())
    weight_hash=hashlib.sha256();shared_hash=hashlib.sha256();capacity_base_hash=hashlib.sha256()
    for name,value in sorted(student.core.state_dict().items()):
        weight_hash.update(name.encode());weight_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        if not name.startswith('coarse_context.'):
            capacity_base_hash.update(name.encode());capacity_base_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        if not name.startswith('observation.'):
            shared_hash.update(name.encode());shared_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    report=dict(status='PASSED',frames=records,original_spatial_calls=calls,
                variant=student.variant,memory_policy=student.memory_policy,core_state_sha256=weight_hash.hexdigest(),
                current_branch_normalized=bool(False),
                trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
                shared_non_observation_sha256=shared_hash.hexdigest(),
                observation_blocks=len(student.core.observation) if isinstance(student.core.observation,torch.nn.Sequential) else 1,
                trainable_by_component={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
                                        for name,module in student.core.named_children()},
                core_conv_macs=conv_macs,core_conv_macs_total=sum(conv_macs.values()),
                cost_scope='One real frame, core Conv2d only; excludes Pillar, warp, normalization, activation, interpolation and data movement; not measured latency.',
                state_bytes=info['state_bytes'],update_radius=2,
                caveat='New model: finite predictions/shape/causality, not parity with the teacher. Crop equivalence is a unit test, not an active sparse policy.')
    if pyramid:
        base_hash=hashlib.sha256()
        for name,value in sorted(student.core.state_dict().items()):
            if not name.startswith('temporal_gates.'):
                base_hash.update(name.encode());base_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        report['pyramid_base_sha256']=base_hash.hexdigest()
        equality=[]
        for target,source in student.migrated_source_pairs(teacher):
            a,b=target.state_dict(),source.state_dict()
            equality.append(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a))
        if initial_pyramid:assert all(equality)
        report.update(mature_spatial_modules_equal_to_source=all(equality),
            spatial_identity_frames=spatial_parity,initial_spatial_identity=initial_pyramid,
            encoder_blocks=[len(student.core.observation),len(student.core.coarse_encoder),len(student.core.deep_encoder)],
            state_shapes={key:list(state[key].shape) for key in student.state_keys},
            cost_scope='One real frame core Conv2d plus ConvTranspose2d dense MAC*2; excludes Pillar, warp, normalization, activation and data movement; not latency.',
            caveat='Complete mature spatial path with three new recurrent states; spatial-only initial parity does not imply INT GRU temporal parity. All modules execute densely.')
    if student.variant == 'two_scale_medium_context_v1':
        report['capacity_base_state_sha256']=capacity_base_hash.hexdigest()
        report['context_projection_initially_zero']=bool(
            torch.count_nonzero(student.core.coarse_context.projection.weight)==0 and
            torch.count_nonzero(student.core.coarse_context.projection.bias)==0)
        report['capacity_initial_exact_parity_frames']=capacity_parity
    # Discard a real supervised optimizer probe before any formal training.
    import copy
    initial=copy.deepcopy(student.state_dict())
    optimizer=make_optimizer(student,optimizer_profile)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    selected=None
    for start in range(len(rows)-3):
        chosen=rows[start:start+4]
        if any(r['supervised'] for r in (chosen[1:] if pyramid else chosen)) and window_ends(chosen,4)=={3}:
            selected=chosen;break
    if selected is None:raise ValueError('No supervised contiguous engineering window')
    state=None;terms=[]
    _,stream=loader(selected,cfg,0,303)
    for sample in stream:
        ego=to_device(sample['ego'],torch.device('cuda'))
        pred,state,_=student.step(ego,state,sample['meta'],detach_state=False)
        if sample['meta']['supervised']:terms.append(criterion(pred,ego['label_dict']))
    loss=torch.stack(terms).mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(student.core.parameters(),5.)
    assert torch.isfinite(norm) and norm>0
    if student.variant == 'two_scale_medium_context_v1':
        gradient=student.core.coarse_context.projection.weight.grad
        report['context_projection_probe_gradient_l1']=float(gradient.abs().sum())
        assert torch.isfinite(gradient).all() and report['context_projection_probe_gradient_l1']>0
    if pyramid:
        report['state_history_probe_gradient_l1']={name:float(getattr(student.core,name).history.weight.grad.abs().sum())
            for name in ('fine_update','coarse_update','deep_update')}
        history_gradients=report['state_history_probe_gradient_l1'].values()
        # Reset intervention supplies zero history and validity at every scale.
        # Its history projection must have zero gradient, unlike recurrence.
        if student.memory_policy == 'reset':
            assert all(value==0 for value in history_gradients)
        else:
            assert all(value>0 and np.isfinite(value) for value in history_gradients)
    optimizer.step()
    assert any(not torch.equal(initial[k],v) for k,v in student.state_dict().items())
    if isinstance(student,GatedPyramidStateDetector) and student.memory_policy=='recurrent':
        # At identity initialization history projection is zero, so gates begin
        # receiving gradients only after the history projection's first update.
        optimizer.zero_grad(set_to_none=True);state=None;terms=[]
        _,stream=loader(selected,cfg,0,303)
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'))
            pred,state,_=student.step(ego,state,sample['meta'],detach_state=False)
            if sample['meta']['supervised']:terms.append(criterion(pred,ego['label_dict']))
        torch.stack(terms).mean().backward()
        report['temporal_gate_second_probe_gradient_l1']={key:float(gate.net[-1].weight.grad.abs().sum())
            for key,gate in student.core.temporal_gates.items()}
        assert all(v>0 and np.isfinite(v) for v in report['temporal_gate_second_probe_gradient_l1'].values())
    student.load_state_dict(initial);student.zero_grad(set_to_none=True)
    assert all(torch.equal(initial[k],v) for k,v in student.state_dict().items())
    report.update(real_supervised_probe_loss=float(loss.detach()),gradient_norm=float(norm),probe_restored=True,
                  probe_optimizer_profile=optimizer_profile)
    if pyramid:
        report.update(engineering_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      engineering_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(out/'engineering.json',report)


def validate(a,student,teacher,cfg,rows,directory):
    a.stage='full'
    student.eval()
    torch.cuda.reset_peak_memory_stats()
    report=evaluate(a,teacher,cfg,rows,directory,student)
    split=getattr(a,'evaluation_split','val')
    report['scope']='Two-scale dense-update full validation' if split=='val' else 'Full training-split fit diagnostic; not validation AP'
    report['evaluation_split']=split
    report['variant']=student.variant;report['memory_policy']=student.memory_policy
    if isinstance(student,PyramidStateDetector):
        report['scope']='Three-scale state pyramid full validation' if split=='val' else 'Full training-split fit diagnostic; not validation AP'
    report['caveats']=[
        'All state cells are updated each frame. No selective-update or sparsity speedup is claimed.',
        'Frozen Pillar; migrated shallow/coarse residual blocks, two-scale states and new spatial reader. No original GRU or teacher in inference.',
        'No teacher execution or teacher state in validation/inference. Model timing excludes IO/voxelization/postprocess.',
        'Concurrent training throughput is not a paired latency benchmark. Use independent same-GPU sequential evaluation after training.']
    if isinstance(student,PyramidStateDetector):
        report['caveats'][1]='Frozen Pillar; complete mature 3/4/5-block encoders and multiscale spatial readout; three new recurrent states. No original GRU or teacher in inference.'
    if split=='train':
        report['caveats'].append('Fixed-weight evaluation on seen training data, without optimizer updates; train/val distribution differences prevent assigning a gap to overfitting alone.')
    for key in list(report):
        if key.startswith('model_step_ms'): report[key]=None
    report['timing_valid']=False
    return report


def train_epoch(a,student,teacher,cfg,rows,optimizer,criterion,epoch,directory,out):
    student.train();teacher.eval();state=teacher_state=None
    ends=window_ends(rows,4);distilled=[];supervised=[];losses=[];labels=updates=0
    gt_windows=[];kd_windows=[]
    _,stream=loader(rows,cfg,4,303+epoch)
    optimizer.zero_grad(set_to_none=True)
    with (directory/'train_frames.jsonl').open('x',buffering=1) as log:
        for index,(row,sample) in enumerate(zip(rows,stream)):
            meta=sample['meta'];assert row['frame']==meta['frame']
            ego=to_device(sample['ego'],torch.device('cuda'))
            # The teacher is an independent frozen stream and never seeds student state.
            with torch.no_grad():target,teacher_state,_=teacher.step(ego,teacher_state,meta,reset=meta['reset_before'])
            prediction,state,info=student.step(ego,state,meta,detach_state=False)
            kd=foreground_head_loss(prediction,target)
            if not bool(torch.isfinite(kd)):raise RuntimeError('Non-finite distillation')
            distilled.append(kd)
            record=dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],reset=info['reset'])
            if meta['supervised']:
                detection=criterion(prediction,ego['label_dict'])
                if not bool(torch.isfinite(detection)):raise RuntimeError('Non-finite supervised detection loss')
                supervised.append(detection);labels+=1
            if index in ends:
                kd_mean=torch.stack(distilled).mean();loss=.2*kd_mean
                gt_mean=torch.stack(supervised).mean() if supervised else None
                if gt_mean is not None:loss=loss+gt_mean
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(student.core.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('Non-finite unified gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()));updates+=1
                record.update(window_end=True,loss=losses[-1],gradient_norm=float(norm))
                if a.variant in ('v2','v2n'):
                    kd_windows.append(float(kd_mean.detach()))
                    if gt_mean is not None:gt_windows.append(float(gt_mean.detach()))
                    record.update(kd_loss=kd_windows[-1],gt_loss=float(gt_mean.detach()) if gt_mean is not None else None)
                state=detach_stream_state(state);distilled=[];supervised=[]
            log.write(json.dumps(record)+'\n')
            if (index+1)%200==0:
                status=dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,total_scans=len(rows),
                            labels=labels,updates=updates,mean_loss=float(np.mean(losses)) if losses else None)
                write_json(out/'status.json',status);print('TRAIN',json.dumps(status),flush=True)
    assert index+1==len(rows) and labels==sum(r['supervised'] for r in rows)
    assert updates==len(ends) and not distilled and not supervised
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.pillar_vfe.parameters())
    result=dict(scans=len(rows),labels=labels,updates=updates,loss=float(np.mean(losses)))
    if a.variant in ('v2','v2n'):result.update(mean_labelled_window_gt=float(np.mean(gt_windows)),mean_window_kd=float(np.mean(kd_windows)))
    return result


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
        oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    teacher,cfg,val,provenance=setup(a)
    for p in teacher.parameters():p.requires_grad_(False)
    write_json(out/'provenance.json',provenance)
    runtime=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train_rows=relocate_tree(json.loads(Path(artifact_path(runtime['manifest'])).read_text())['train'])
    torch.manual_seed(303);np.random.seed(303);random.seed(303)
    student=(GatedPyramidStateDetector(teacher,cfg,a.memory_policy) if a.variant=='pyramid_gated' else
             PyramidStateDetector(teacher,cfg,a.memory_policy) if a.variant=='pyramid_large'
             else TwoScaleDetector(teacher,cfg,a.memory_policy,a.variant)).cuda()
    a.engineering=str(out/'engineering.json')
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location='cpu')
        assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
        assert saved['contract']['data_contract']==provenance['contract']
        assert saved['contract']['version']==student.version
        assert saved['contract'].get('memory_policy','recurrent')==a.memory_policy
        student.load_state_dict(saved['model'],strict=True)
        # Keep checkpoint identity separate from the inference-only intervention.
        student.memory_policy=a.evaluation_memory_policy or a.memory_policy
        evaluation_rows=train_rows if a.evaluation_split=='train' else val
        provenance['split']=a.evaluation_split
        write_json(out/'provenance.json',provenance)
        engineering(student,teacher,cfg,evaluation_rows,out,a.optimizer_profile)
        if a.diagnostic:
            from opencood.tools.diagnose_int_unified_bev import diagnose
            report=diagnose(student,teacher,cfg,val,out)
        else:
            report=validate(a,student,teacher,cfg,evaluation_rows,out)
        report.update(checkpoint=a.checkpoint,checkpoint_sha256=digest(Path(a.checkpoint)))
        report.update(training_memory_policy=a.memory_policy,
                      evaluation_memory_policy=student.memory_policy,
                      training_optimizer_profile=saved['contract'].get('optimizer_profile','uniform'))
        if student.memory_policy != a.memory_policy:
            report['caveats'].append('Memory policy is an inference-only intervention on this exact checkpoint; not a separately trained baseline.')
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(status='COMPLETED'))
        return
    contract=dict(version='two_scale_dense_v1',seed=303,epochs_min=5,epochs_max=a.epochs,
        data_contract=provenance['contract'],teacher_sha256=provenance['checkpoint']['sha256'],
        train_scans=len(train_rows),val_scans=len(val),tbptt=4,patience=5,min_delta=.001,
        optimizer='AdamW',initial_lr=.0003,weight_decay=.0001,distillation_weight=.2,
        trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in student.parameters() if not p.requires_grad),
        architecture='Pillar64 frozen; observation96 stride2; coarse context32 at stride4; single128 state; radius2 gated update; pointwise256 readout and trainable detection heads',
        loss='Mean GT detection loss over labelled frames + .2 mean foreground head distillation over all frames in each TBPTT window; no teacher feature target',
        protocol='Dense two-scale autoregressive states; student state only; no augmentation/delay; scene/gap>.25s resets; no teacher refresh',
        future_sparse='Radius2 crop parity only; no learned selector, skipped state cells or coarse-context caching in this baseline')
    if a.variant in ('v2','v2n'):
        contract.update(version='unified_bev_dense_'+a.variant,memory_policy=a.memory_policy,
            architecture='Pillar64 frozen; residual observation128 stride2; two downsample context stages192/256 with dilated residual blocks; single256 state; radius2 residual gated update with cellwise channel normalization; pointwise256 readout and trainable detection heads',
            protocol='Dense every-frame state update; memory_policy='+a.memory_policy+'; no augmentation/delay; same segment/gap resets and TBPTT windows as v1; no teacher state or periodic refresh',
            future_sparse='Radius2 update crop parity only, given full precomputed evidence; observation/context/readout remain dense; no active selector')
    if a.variant=='v2n':
        contract['current_branch']='Non-affine per-cell channel LayerNorm after current projection, before residual addition; no added parameters; all other V2 architecture/initialization unchanged'
        contract['architecture']+='; normalize current shortcut with non-affine LayerNorm'
    contract.update(version='two_scale_dense_v1', memory_policy=a.memory_policy, initial_lr=.0003,
        architecture='Mature first128 block; two coarse256 residual blocks at original stride; fine128/coarse256 explicit states; bottleneck history updates; spatial256 readout. No FM/teacher in inference.',
        future_sparse='Not implemented; dense controls first', normalization='Frozen BN statistics, trainable affine')
    if a.variant != 'two_scale_v1':
        contract.update(version=student.version,
            architecture=('Full mature fine encoder' if a.variant in ('two_scale_full','two_scale_medium') else 'First mature fine block')+
                '; two coarse256 residual blocks at original stride; fine128/coarse256 explicit states; unchanged bottleneck history updates; pointwise256 fusion plus radius2 width64 residual reader; no FM/teacher in inference.')
    if a.variant == 'two_scale_medium':
        contract['architecture']+=' Stateless stride8 context: 256->384 stride2 3x3, three width192 residual bottlenecks, zero-initialized 384->256 projection and nearest resize added before coarse update.'
        contract['capacity_control']='two_scale_full; all shared initial weights and initial outputs match; same two recurrent state shapes; added context is evaluated densely every frame'
    if a.variant in ('pyramid_large','pyramid_gated'):
        contract.update(version=student.version,
            architecture='Complete mature 3/4/5-block 128/256/512 encoders, three transposed-convolution readout branches, full mature spatial readout and heads; new sequential pose-aligned states at strides2/4/8; no original input GRU or teacher in inference.',
            priority='Detection AP first; no small-parameter, low-resolution-only or local-crop constraint',
            protocol='Dense three-scale autoregressive states; student state only; no augmentation/delay; scene/gap>.25s resets; TBPTT4 detaches all three states; no teacher refresh',
            capacity_control='Compare with INT and smaller state models; architecture, state layout and capacity jointly change, not parameter-only ablation')
    if a.variant == 'pyramid_gated':
        contract['architecture']+=' Per-scale channel/spatial gate on aligned history: concat(current,history,difference,valid), width C/2, 1x1 and 3x3 with GroupNorm, zero-ended 1x1, multiplier 2*sigmoid; initially one.'
        contract['capacity_control']='pyramid_large transfer; shared initial weights and initial predictions identical; gate adds capacity and temporal weighting jointly; same three states and training recipe'
    optimizer=make_optimizer(student,a.optimizer_profile)
    min_lrs=([1e-5] if a.optimizer_profile=='uniform' else [group['lr']/30 for group in optimizer.param_groups])
    if a.optimizer_profile != 'uniform':
        contract.update(optimizer_profile=a.optimizer_profile,
            optimizer_groups=[dict(name=g['name'],initial_lr=g['lr'],min_lr=minimum,
                parameters=sum(p.numel() for p in g['params']))
                for g,minimum in zip(optimizer.param_groups,min_lrs)])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,
        threshold=.001,threshold_mode='abs',min_lr=min_lrs)
    best=significant=-1.;stale=0;history=[];start=1
    if a.resume:
        source=Path(a.resume).parent
        assert json.loads((source/'status.json').read_text())['status']=='COMPLETED'
        assert json.loads((source/'report.json').read_text())['stop_reason']=='max_epochs_not_proof_of_convergence'
        saved=torch.load(a.resume,map_location='cpu')
        old=dict(saved['contract']);new=dict(contract);old.pop('epochs_max');new.pop('epochs_max')
        assert old==new and a.epochs>saved['epoch']
        assert saved['history']==json.loads((source/'history.json').read_text())
        student.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        history=saved['history'];best=saved['best'];significant=saved['significant'];stale=saved['stale'];start=saved['epoch']+1
        prior_best=json.loads((source/'best.json').read_text())
        assert prior_best['ap70']==best and digest(Path(prior_best['checkpoint']))==prior_best['sha256']
        write_json(out/'best.json',prior_best);write_json(out/'history.json',history)
        write_json(out/'resumption.json',dict(source=str(source),checkpoint_sha256=digest(Path(a.resume)),next_epoch=start))
    write_json(out/'training_contract.json',contract)
    engineering(student,teacher,cfg,train_rows,out,a.optimizer_profile)
    if a.engineering_only:
        write_json(out/'status.json',dict(status='COMPLETED',scope='engineering only'));return
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    for epoch in range(start,a.epochs+1):
        tick=time.monotonic();directory=out/('epoch_%03d'%epoch);directory.mkdir()
        result=train_epoch(a,student,teacher,cfg,train_rows,optimizer,criterion,epoch,directory,out)
        path=directory/'student.pth'
        torch.save(dict(model=student.state_dict(),epoch=epoch,contract=contract),str(path))
        write_json(out/'status.json',dict(status='RUNNING',phase='full_validation',epoch=epoch))
        report=validate(a,student,teacher,cfg,val,directory)
        report.update(epoch=epoch,checkpoint_sha256=digest(path));write_json(directory/'report.json',report)
        ap=report['metrics']['bev_ap70'];scheduler.step(ap)
        if ap>best:
            best=ap;write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(path),sha256=digest(path)))
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,ap70=ap,stale_epochs=stale,lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-tick,**result))
        if a.optimizer_profile != 'uniform':
            history[-1]['group_lrs']={g['name']:g['lr'] for g in optimizer.param_groups}
        write_json(out/'history.json',history)
        save_checkpoint(out/'latest.pth',dict(model=student.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            contract=contract,epoch=epoch,history=history,best=best,significant=significant,stale=stale,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
        if epoch>=5 and stale>=5 and optimizer.param_groups[0]['lr']<=.0001:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),
        stop_reason='plateau' if stale>=5 and optimizer.param_groups[0]['lr']<=.0001 else 'max_epochs_not_proof_of_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=['two_scale_v1','two_scale_narrow','two_scale_full','two_scale_medium','pyramid_large','pyramid_gated'],default='two_scale_v1')
    p.add_argument('--memory-policy',choices=['recurrent','reset'],default='recurrent')
    p.add_argument('--optimizer-profile',choices=['uniform','transfer','protect_spatial'],default='uniform')
    p.add_argument('--evaluation-memory-policy',choices=['recurrent','reset'],
                   help='Override only full checkpoint evaluation; --memory-policy must still match its training contract')
    p.add_argument('--evaluation-split',choices=['val','train'],default='val',
                   help='Train is a fixed-checkpoint fit diagnostic only; never used for training-time model selection')
    p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--diagnostic',action='store_true',help='256-frame teacher/student/reset diagnostic; requires checkpoint')
    a=p.parse_args();out=Path(a.output)
    if a.epochs<5 or (a.checkpoint and a.resume):p.error('At least five epochs; evaluation and resume are exclusive')
    if a.diagnostic and not a.checkpoint:p.error('Diagnostic requires an immutable checkpoint')
    if a.evaluation_memory_policy and (not a.checkpoint or a.resume or a.diagnostic or a.engineering_only):
        p.error('Evaluation memory override requires full --checkpoint evaluation only')
    if a.evaluation_split=='train' and (not a.checkpoint or a.diagnostic):
        p.error('Training-split evaluation requires --checkpoint and cannot combine with --diagnostic')
    if a.variant=='v1' and a.memory_policy!='recurrent':p.error('Reset-policy training is a v2 control')
    if a.worker:
        def expired(signum, frame): raise TimeoutError('24h hard budget')
        signal.signal(signal.SIGALRM,expired);signal.alarm(86400)
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<12000:raise RuntimeError('Insufficient free GPU memory')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.run_int_two_scale']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=process.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid,output=str(out))))


if __name__=='__main__':main()

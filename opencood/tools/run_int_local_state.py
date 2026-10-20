"""Train the approved migrated local-state baseline, before region skipping."""
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
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_local_state import LocalStateDetector, feature_distillation
from opencood.models.sub_modules.int_local_state_context import ContextLocalStateDetector, load_refinement_state
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.tools.run_int_cache_repair import setup, evaluate
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def teacher_step(teacher, data, state, meta):
    captured = []
    hook = teacher.cls_head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    try:
        with torch.no_grad():
            prediction, state, _ = teacher.step(data, state, meta, reset=meta.get('reset_before', False))
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one teacher detection feature')
    return prediction, captured[0], state


def make_optimizer(student, refinement=False, head_lr=None):
    if head_lr is not None and (not refinement or not 0 < head_lr <= 1e-3):
        raise ValueError('Head LR override requires refinement and a finite rate in (0, 1e-3]')
    heads = list(student.cls_head.parameters()) + list(student.reg_head.parameters())
    if student.dir_head is not None:
        heads += list(student.dir_head.parameters())
    groups = [
        dict(params=student.update.input_projection.parameters(), lr=1e-4 if refinement else 1e-3, name='projection'),
        dict(params=student.update.blocks[:2].parameters(), lr=1e-5 if refinement else 1e-4, name='migrated_update'),
        dict(params=student.observation.parameters(), lr=1e-6 if refinement else 1e-5, name='shallow'),
        dict(params=heads, lr=head_lr if head_lr is not None else (1e-6 if refinement else 1e-5), name='heads')]
    if len(student.update.blocks) > 2:
        groups.append(dict(params=student.update.blocks[2:].parameters(), lr=1e-4, name='new_context'))
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def step_losses(student, teacher, ego, meta, state, teacher_state, criterion):
    target, features, teacher_state = teacher_step(teacher, ego, teacher_state, meta)
    prediction, state, info = student.step(ego, state, meta, detach_state=False)
    detection = criterion(prediction, ego['label_dict']) if meta['supervised'] else None
    head = foreground_head_loss(prediction, target)
    feature = feature_distillation(state['value'], features, target)
    if not all(bool(torch.isfinite(x)) for x in [head, feature] + ([] if detection is None else [detection])):
        raise RuntimeError('Nonfinite local-state losses')
    return detection, head, feature, state, teacher_state, info


def check_migration(student, teacher):
    # Source weights and shape must match before the first optimization step.
    mapping = [('pillar_vfe', teacher.pillar_vfe), ('observation', teacher.backbone.resnet.layer0),
               ('cls_head', teacher.cls_head), ('reg_head', teacher.reg_head)]
    if teacher.use_dir:
        mapping.append(('dir_head', teacher.dir_head))
    for name, source in mapping:
        for key, value in getattr(student, name).state_dict().items():
            assert torch.equal(value.cpu(), source.state_dict()[key].cpu()), (name, key)
    for index in range(2):
        source = teacher.backbone.resnet.layer1[index].state_dict()
        assert all(torch.equal(value.cpu(), source[key].cpu()) for key, value in student.update.blocks[index].state_dict().items())
    assert student.update.blocks[0].conv1.stride == (1, 1)
    assert teacher.backbone.resnet.layer1[0].conv1.stride == (2, 2)
    assert not hasattr(student, 'feature_memory') and not hasattr(student, 'backbone') and not hasattr(student, 'shrink_conv')


def engineering(a, student, teacher, cfg, rows, out, reference=None):
    # For B this snapshot must be the named A checkpoint, not raw migration.
    initial = copy.deepcopy(student.state_dict())
    criterion = PointPillarDirLoss(cfg['loss']['args']).to(a.device)
    student.train()
    optimizer = make_optimizer(student, bool(a.refine_from), a.refine_head_lr)
    optimizer_groups = [dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params']))
                        for g in optimizer.param_groups]
    grouped_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(p) for p in student.parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    state = teacher_state = None
    terms, records, initial_errors = [], [], []
    reference_state = None
    first_label = next((i for i, row in enumerate(rows) if row['supervised']), None)
    if first_label is None:
        raise ValueError('No supervised frame in the training manifest')
    start = max(0, first_label - a.frames + 1)
    chosen = rows[start:start+a.frames]
    if a.phase == 'b':
        # A real uninterrupted four-frame graph, including a supervised target.
        for start in range(len(rows)-a.frames+1):
            chosen = rows[start:start+a.frames]
            if any(r['supervised'] for r in chosen) and window_ends(chosen, a.frames) == {a.frames-1}:
                break
        else:
            raise ValueError('No uninterrupted supervised engineering window')
    _, stream = loader(chosen, cfg, 0, 303)
    for sample in stream:
        meta = sample['meta']
        ego = to_device(sample['ego'], torch.device(a.device))
        gt, head, feature, state, teacher_state, info = step_losses(student, teacher, ego, meta, state, teacher_state, criterion)
        if reference is not None:
            with torch.no_grad():
                expected, reference_state, _ = reference.step(ego, reference_state, meta)
                actual = dict(psm=student.cls_head(state['value']), rm=student.reg_head(state['value']))
                if student.dir_head is not None:
                    actual['dm'] = student.dir_head(state['value'])
                error = max(float((actual[k]-expected[k]).abs().max()) for k in expected)
                initial_errors.append(error)
                assert error <= 1e-5, 'Refinement must preserve source predictions before training'
        terms.append((gt if gt is not None else 0) + .2 * head + .5 * feature)
        assert state['value'].shape[-2:] == ego['anchor_box'].shape[:2]
        records.append(dict(frame=meta['frame'],supervised=meta['supervised'],gt=None if gt is None else float(gt.detach()),
                            head_kd=float(head.detach()),feature_kd=float(feature.detach()),state_shape=list(state['value'].shape),
                            reset=info['reset']))
        write_json(out/'status.json',dict(status='RUNNING',phase='engineering',frames=len(records),total=a.frames))
    assert any(r['supervised'] for r in records), 'Engineering prefix must include labelled data'
    loss = torch.stack(terms).mean()
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(student.parameters(),5.)
    assert torch.isfinite(gradient) and gradient > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.pillar_vfe.parameters())
    history_gradient = float(student.update.input_projection.weight.grad[:,128:384].abs().sum())
    context_gradients = [float(block.bn2.weight.grad.abs().sum()) for block in student.update.blocks[2:]]
    assert all(value > 0 for value in context_gradients)
    if a.phase == 'b':
        assert all(r['reset'] is None for r in records[1:]), 'B probe must reuse its own state'
        assert history_gradient > 0, 'History projection must learn from the real sequence'
    optimizer.step()
    changed = not torch.equal(initial['update.input_projection.weight'], student.update.input_projection.weight.detach())
    assert changed
    temporal_gradient = None
    if a.phase == 'b':
        # At A initialization history weights are zero. After one discarded
        # probe update, verify the last loss reaches a previous frame's state.
        optimizer.zero_grad(set_to_none=True)
        state = teacher_state = None
        states = []
        _, stream = loader(chosen, cfg, 0, 303)
        for sample in stream:
            ego = to_device(sample['ego'], torch.device(a.device))
            gt, head, feature, state, teacher_state, _ = step_losses(
                student, teacher, ego, sample['meta'], state, teacher_state, criterion)
            state['value'].retain_grad()
            states.append(state['value'])
        final_loss = (gt if gt is not None else 0) + .2 * head + .5 * feature
        final_loss.backward()
        temporal_gradient = sum(float(v.grad.abs().sum()) for v in states[:-1] if v.grad is not None)
        assert np.isfinite(temporal_gradient) and temporal_gradient > 0
    student.load_state_dict(initial, strict=True)
    student.zero_grad(set_to_none=True)
    student.eval()
    assert all(torch.equal(initial[k], value) for k, value in student.state_dict().items())
    return dict(status='PASSED',device=a.device,frames=records,manifest_start_index=start,
                selection='First contiguous requested-length window containing a training label; initial state empty',
                gradient_norm_before_clip=float(gradient),
                optimizer_step_changed_projection=changed,initial_state_restored=True,
                initialization='fixed_phase_b_refinement' if a.refine_from else
                    ('named_phase_a_checkpoint' if a.phase == 'b' else 'raw_teacher_migration'),
                source_prediction_max_errors=initial_errors,
                optimizer_groups=optimizer_groups,
                new_context_bn2_gradient_l1=context_gradients,
                history_projection_gradient_l1=history_gradient,
                previous_state_gradient_l1_after_probe_update=temporal_gradient,
                trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in student.parameters() if not p.requires_grad),
                migrated_tensor_equality=True,migration_check_scope='Before optional checkpoint loading',update_radius=student.update.radius,
                caveat='Engineering only; no AP or speed claim. Probe optimizer step is discarded before formal training.')


def train_epoch(a, student, teacher, cfg, rows, optimizer, criterion, epoch, directory, out):
    student.train()
    student.set_head_trainable(a.phase == 'b' or epoch > 1)
    teacher.eval()
    ends = window_ends(rows, 4)
    state = teacher_state = None
    gt_terms, head_terms, feature_terms = [], [], []
    losses, gts, heads, features = [], [], [], []
    labels = 0
    optimizer.zero_grad(set_to_none=True)
    _, stream = loader(rows, cfg, 4, 303 + epoch)
    with (directory/'train_frames.jsonl').open('x', buffering=1) as log:
        for index, (row, sample) in enumerate(zip(rows, stream)):
            meta = sample['meta']
            assert row['frame'] == meta['frame']
            ego = to_device(sample['ego'], torch.device(a.device))
            gt, head, feature, state, teacher_state, info = step_losses(student,teacher,ego,meta,state,teacher_state,criterion)
            if gt is not None:
                gt_terms.append(gt)
                labels += 1
            head_terms.append(head)
            feature_terms.append(feature)
            record = dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],reset=info['reset'])
            if index in ends:
                h, f = torch.stack(head_terms).mean(), torch.stack(feature_terms).mean()
                g = torch.stack(gt_terms).mean() if gt_terms else None
                loss = .2 * h + .5 * f + (g if g is not None else 0)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(),5.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError('Nonfinite local-state gradients')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach())); heads.append(float(h.detach())); features.append(float(f.detach()))
                if g is not None:
                    gts.append(float(g.detach()))
                record.update(window_end=True,loss=losses[-1],head_kd=heads[-1],feature_kd=features[-1],
                              gt_loss=None if g is None else float(g.detach()),gradient_norm=float(norm))
                state = dict(state,value=state['value'].detach())
                gt_terms, head_terms, feature_terms = [], [], []
            log.write(json.dumps(record)+'\n')
            if (index+1) % 200 == 0:
                progress = dict(status='RUNNING',phase='train_'+a.phase,epoch=epoch,scans=index+1,total_scans=len(rows),
                                labels=labels,updates=len(losses),mean_loss=float(np.mean(losses)))
                write_json(out/'status.json',progress)
                print('TRAIN',json.dumps(progress),flush=True)
    assert index+1 == len(rows) and labels == sum(r['supervised'] for r in rows)
    assert len(losses) == len(ends) and not head_terms
    return dict(scans=len(rows),labels=labels,updates=len(losses),loss=float(np.mean(losses)),
                gt_loss=float(np.mean(gts)),head_kd=float(np.mean(heads)),feature_kd=float(np.mean(features)))


def save_checkpoint(path, saved):
    temporary = str(path) + '.tmp'
    torch.save(saved, temporary)
    os.replace(temporary, str(path))


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,max_age=.25,
                           oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():
        setattr(a,key,value)
    teacher,cfg,val,provenance = setup(a)
    for p in teacher.parameters():
        p.requires_grad_(False)
    write_json(out/'provenance.json',provenance)
    runtime = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train = json.loads(Path(runtime['manifest']).read_text())['train']
    torch.manual_seed(303); np.random.seed(303)
    factory = ContextLocalStateDetector if a.variant == 'context' else LocalStateDetector
    student = factory(teacher,cfg,'reset' if a.phase=='a' else 'recurrent').to(a.device)
    tests = subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:
        raise RuntimeError('INT engineering tests failed')
    check_migration(student,teacher)
    contract = dict(version=student.version,phase=a.phase,memory_policy=student.memory_policy,
                    teacher_sha256=provenance['checkpoint']['sha256'],data_contract=provenance['contract'],
                    seed=303,tbptt=4,train_scans=len(train),train_labels=sum(r['supervised'] for r in train),
                    val_scans=len(val),val_labels=sum(r['supervised'] for r in val),epochs_max=a.epochs,
                    loss_weights=dict(gt=1.,head_kd=.2,feature_kd=.5),feature_loss='channel RMS clamp1; .5 global + .5 confidence weighted SmoothL1',
                    optimizer='AdamW',initial_lrs=[g['lr'] for g in make_optimizer(student,bool(a.refine_from),a.refine_head_lr).param_groups],
                    refinement_head_lr_override=a.refine_head_lr,
                    weight_decay=1e-4,clip_gradient=5.,
                    head_warmup_epochs=1 if a.phase=='a' else 0,normalization='fixed BN statistics; affine trainable',
                    update_radius=student.update.radius,all_regions_update=True,augmentation=False,artificial_delay=False)
    reference = None
    if a.initialize_from:
        source = Path(a.initialize_from)
        saved = torch.load(str(source),map_location='cpu')
        old = saved['contract']
        assert old['version']==student.version and old['phase']=='a' and a.phase=='b'
        assert old['teacher_sha256']==contract['teacher_sha256'] and old['data_contract']==contract['data_contract']
        student.load_state_dict(saved['model'],strict=True)
        contract['initialization'] = dict(path=str(source),sha256=digest(source),epoch=saved['epoch'])
    if a.refine_from:
        source = Path(a.refine_from)
        assert digest(source) == a.source_sha256, 'Fixed refinement checkpoint hash mismatch'
        saved = torch.load(str(source), map_location='cpu')
        old = saved['contract']
        assert old['version'] == LocalStateDetector.version and old['phase'] == 'b'
        assert old['teacher_sha256'] == contract['teacher_sha256'] and old['data_contract'] == contract['data_contract']
        load_refinement_state(student, saved['model'])
        reference = LocalStateDetector(teacher,cfg,'recurrent').to(a.device).eval()
        reference.load_state_dict(saved['model'],strict=True)
        contract.update(refinement_variant=a.variant, initialization=dict(
            path=str(source),sha256=a.source_sha256,epoch=saved['epoch'],source_phase='b'),
            epoch_budget_is_hard=True, plateau_projection_lr_limit=1e-5)
    write_json(out/'training_contract.json',contract)
    gate = engineering(a,student,teacher,cfg,train,out,reference)
    write_json(out/'engineering.json',gate)
    del reference
    if a.engineering_only:
        write_json(out/'report.json',dict(status='ENGINEERING_PASSED',validation_ap=None,
                                         scope='Real-data losses/gradient/migration only; not training or convergence'))
        write_json(out/'status.json',dict(status='COMPLETED'))
        return
    torch.manual_seed(303); np.random.seed(303)
    optimizer = make_optimizer(student,bool(a.refine_from),a.refine_head_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,
                                                         threshold=.001,threshold_mode='abs',min_lr=1e-7)
    history=[]; best=significant=-1.; stale=0
    criterion = PointPillarDirLoss(cfg['loss']['args']).to(a.device)
    a.engineering = str(out/'engineering.json')
    for epoch in range(1,a.epochs+1):
        tick=time.monotonic(); directory=out/('epoch_%03d'%epoch); directory.mkdir()
        result=train_epoch(a,student,teacher,cfg,train,optimizer,criterion,epoch,directory,out)
        path=directory/'student.pth'
        save_checkpoint(path,dict(model=student.state_dict(),epoch=epoch,contract=contract))
        student.eval(); a.stage='full'; torch.cuda.reset_peak_memory_stats()
        write_json(out/'status.json',dict(status='RUNNING',phase='full_validation',epoch=epoch))
        report=evaluate(a,teacher,cfg,val,directory,student)
        report.update(scope='Migrated local-state dense baseline phase '+a.phase,epoch=epoch,
                      checkpoint_sha256=digest(path),memory_policy=student.memory_policy,
                      caveats=['All regions update; no region skipping or speedup claim.',
                               'Teacher used only in training; validation owns student state.',
                               'Concurrent GPU timing is not a controlled latency comparison.'])
        write_json(directory/'report.json',report)
        ap=report['metrics']['bev_ap70']; scheduler.step(ap)
        if ap>best:
            best=ap; write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(path),sha256=digest(path)))
        if ap>significant+.001:significant=ap; stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,ap70=ap,stale_epochs=stale,lrs=[g['lr'] for g in optimizer.param_groups],
                            seconds=time.monotonic()-tick,**result))
        write_json(out/'history.json',history)
        save_checkpoint(out/'latest.pth',dict(model=student.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),contract=contract,history=history,epoch=epoch,
            best=best,significant=significant,stale=stale,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()))
        if epoch>=5 and stale>=5 and optimizer.param_groups[0]['lr']<=(1e-5 if a.refine_from else 1e-4):
            break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),
        stop_reason='plateau' if stale>=5 and optimizer.param_groups[0]['lr']<=(1e-5 if a.refine_from else 1e-4) else 'max_epochs_not_proof_of_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True); p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['a','b'],default='a'); p.add_argument('--initialize-from')
    p.add_argument('--refine-from'); p.add_argument('--source-sha256')
    p.add_argument('--refine-head-lr',type=float,
                   help='Explicit head-only learning-rate intervention for fixed-checkpoint refinement')
    p.add_argument('--variant',choices=['base','context'],default='base')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--gpu',type=int)
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--frames',type=int,default=2)
    p.add_argument('--engineering-only',action='store_true'); p.add_argument('--worker',action='store_true')
    a=p.parse_args(); out=Path(a.output)
    if not 2<=a.frames<=8 or not 5<=a.epochs<=60:p.error('Engineering frames 2..8; training epochs 5..60')
    if a.initialize_from and a.refine_from:p.error('Initialization and refinement are exclusive')
    if (a.phase=='b') != bool(a.initialize_from or a.refine_from):p.error('Phase B requires explicit initialization')
    if a.refine_from and (not a.source_sha256 or a.epochs>20):p.error('Refinement requires a fixed SHA and at most 20 epochs')
    if a.refine_head_lr is not None and (not a.refine_from or not 0<a.refine_head_lr<=1e-3):
        p.error('Head LR override requires refinement and a finite rate in (0, 1e-3]')
    if a.variant=='context' and not a.refine_from:p.error('Context variant requires fixed phase-B refinement')
    if a.source_sha256 and not a.refine_from:p.error('Source SHA applies only to refinement')
    if a.phase=='b' and a.frames != 4:p.error('Phase B engineering requires a real four-frame TBPTT window')
    if a.device=='cpu' and not a.engineering_only:p.error('CPU is restricted to bounded engineering')
    if a.device=='cuda' and a.gpu is None:p.error('CUDA requires explicit GPU')
    timeout=1800 if a.engineering_only else (86400 if a.refine_from else 172800)
    if a.worker:
        def expired(signum,frame):raise TimeoutError('Local-state execution budget exceeded')
        signal.signal(signal.SIGALRM,expired); signal.alarm(timeout)
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc))); raise
        finally:signal.alarm(0)
        return
    if a.device=='cuda':
        free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu',
            '--format=csv,noheader,nounits']).decode().strip().split(',')
        if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied; no automatic shared training')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.run_int_local_state']+sys.argv[1:]+['--worker']
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES='' if a.device=='cpu' else str(a.gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (out/'run.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=process.pid,command=command,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),timeout_seconds=timeout,device=a.device))
    print(json.dumps(dict(pid=process.pid,output=str(out))))


if __name__=='__main__':main()

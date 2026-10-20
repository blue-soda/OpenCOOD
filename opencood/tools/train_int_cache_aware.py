"""Matched dense/cache-aware U refinement; frozen observation and tile scorer."""
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
from opencood.models.sub_modules.int_local_selector import selector_state
from opencood.tools.evaluate_int_local_cache import make_model, probe
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_local_state import step_losses, train_epoch, save_checkpoint
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def optimizer_for(model):
    heads = list(model.cls_head.parameters()) + list(model.reg_head.parameters())
    if model.dir_head is not None:
        heads += list(model.dir_head.parameters())
    groups = [dict(params=model.update.input_projection.parameters(), lr=1e-4, name='projection'),
              dict(params=model.update.blocks.parameters(), lr=1e-5, name='update_blocks'),
              dict(params=heads, lr=1e-6, name='heads')]
    result = torch.optim.AdamW(groups, weight_decay=1e-4)
    ids = [id(p) for group in result.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    return result


def engineering(student, reference, teacher, cfg, rows):
    initial = copy.deepcopy(student.state_dict())
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda()
    for start in range(len(rows)-3):
        chosen = rows[start:start+4]
        if any(r['supervised'] for r in chosen) and window_ends(chosen, 4) == {3}:
            break
    else:
        raise ValueError('No contiguous labelled four-frame train window')
    student.train()
    state = teacher_state = ref_state = None
    retained, errors = [], []
    _, stream = loader(chosen, cfg, 0, 303)
    for sample in stream:
        ego = to_device(sample['ego'], torch.device('cuda'))
        gt, head, feature, state, teacher_state, info = step_losses(
            student, teacher, ego, sample['meta'], state, teacher_state, criterion)
        state['value'].retain_grad()
        retained.append(state['value'])
        with torch.no_grad():
            expected, ref_state, _ = reference.step(ego, ref_state, sample['meta'])
            actual = dict(psm=student.cls_head(state['value']), rm=student.reg_head(state['value']))
            if student.dir_head is not None:
                actual['dm'] = student.dir_head(state['value'])
            error = max(float((expected[k]-actual[k]).abs().max()) for k in expected)
            errors.append(error)
            assert all(torch.allclose(expected[k], actual[k], atol=1e-4, rtol=1e-4) for k in expected)
    loss = (gt if gt is not None else 0) + .2*head + .5*feature
    loss.backward()
    temporal = sum(float(x.grad.abs().sum()) for x in retained[:-1] if x.grad is not None)
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.)
    assert torch.isfinite(norm) and norm > 0 and temporal > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.observation.parameters())
    if hasattr(student, 'selector_net'):
        assert all(p.grad is None for p in student.selector_net.parameters())
    optimizer_for(student).step()
    assert not torch.equal(initial['update.input_projection.weight'], student.update.input_projection.weight)
    for name, value in student.state_dict().items():
        if name.startswith(('observation.', 'pillar_vfe.', 'selector_net.')) or 'running_' in name or 'num_batches_tracked' in name:
            assert torch.equal(initial[name], value), name
    student.load_state_dict(initial, strict=True)
    student.zero_grad(set_to_none=True); student.eval()
    assert all(torch.equal(v, initial[k]) for k,v in student.state_dict().items())
    return dict(status='PASSED', train_start_index=start, frames=4,
                prediction_max_errors=errors, previous_state_gradient_l1=temporal,
                gradient_norm=float(norm), discarded_optimizer_step=True,
                initial_weights_restored=True, observation_selector_bn_frozen=True)


def evaluate_models(cfg, rows, models, out, source_ap):
    states = {n:None for n in models}
    aps = {n:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in [.3,.5,.7]} for n in models}
    times = {n:[] for n in models}
    regions = {n:[] for n in models}
    resets = {n:0 for n in models}
    before = {n:{k:v.detach().clone() for k,v in m.state_dict().items()} for n,m in models.items()}
    for model in models.values():
        model.eval()
    dataset, stream = loader(rows, cfg, 4, 303)
    with torch.no_grad(), (out/'eval_frames.jsonl').open('x', buffering=1) as log:
        for index, sample in enumerate(stream):
            assert sample['meta']['frame'] == rows[index]['frame']
            ego = to_device(sample['ego'], torch.device('cuda'))
            names = list(models); names = names[index%len(names):]+names[:index%len(names)]
            predictions = {}
            for name in names:
                torch.cuda.synchronize(); tick = time.perf_counter()
                predictions[name], states[name], info = models[name].step(ego, states[name], sample['meta'])
                torch.cuda.synchronize(); elapsed = (time.perf_counter()-tick)*1000
                assert all(torch.isfinite(v).all() for v in predictions[name].values())
                if index >= 10:
                    times[name].append(elapsed)
                resets[name] += bool(info['reset'])
                if 'region_cache' in info:
                    regions[name].append(info['region_cache'])
                log.write(json.dumps(dict(frame=sample['meta']['frame'], mode=name, ms=elapsed,
                                          reset=info['reset'], region=info.get('region_cache')))+'\n')
            if sample['meta']['supervised']:
                data = dict(ego=ego); gt = dataset.post.generate_gt_bbx(data)
                for name, prediction in predictions.items():
                    boxes, scores = dataset.post.post_process(data, dict(ego=prediction))
                    for t in aps[name]:
                        eval_utils.caluclate_tp_fp(boxes, scores, gt, aps[name], t)
            if (index+1) % 200 == 0:
                write_json(out/'status.json', dict(status='RUNNING', scans=index+1, total=len(rows)))
    assert index+1 == len(rows)
    results = {}
    for name, model in models.items():
        assert all(torch.equal(v, before[name][k]) for k,v in model.state_dict().items())
        results[name] = dict(metrics={'bev_ap%d'%int(t*100):eval_utils.calculate_ap(aps[name],t)[0] for t in aps[name]},
                             mean_ms=float(np.mean(times[name])), p95_ms=float(np.percentile(times[name],95)),
                             reset_frames=resets[name],
                             region_means={k:float(np.mean([r[k] for r in regions[name]]))
                                           for k in regions[name][0]} if regions[name] else None)
    assert abs(results['source_dense']['metrics']['bev_ap70']-source_ap)<1e-8
    report = dict(status='COMPLETED', scans=len(rows), labels=sum(r['supervised'] for r in rows),
                  results=results, validation_weights_unchanged=True,
                  timing_scope='Same GPU/frame, rotating mode order; synchronized model including selector, warps, crops and writeback; excludes IO/voxelization/H2D/postprocess.',
                  caveats=['Training uses full U with masked writeback; only validation uses actual local execution.',
                           'Same-GPU timings are preliminary and include concurrent host contention.',
                           'Fixed three-epoch feasibility budget, not a convergence claim.'])
    write_json(out/'report.json', report)
    write_json(out/'status.json', dict(status='COMPLETED'))
    return report


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,
                    max_age=.25,oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False,
                    phase='b').items():
        setattr(a,k,v)
    teacher,cfg,val,provenance = setup(a)
    for param in teacher.parameters():
        param.requires_grad_(False)
    assert digest(a.checkpoint)==a.source_sha256 and digest(a.selector_checkpoint)==a.selector_sha256
    saved = torch.load(a.checkpoint, map_location='cpu')
    assert saved['contract']['phase']=='b' and saved['contract']['refinement_variant']=='base'
    assert saved['contract']['data_contract']==provenance['contract']
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    selected = torch.load(a.selector_checkpoint, map_location='cpu')
    weights = selector_state(selected,a.source_sha256,16,.5)
    assert selected['max_age']==.5 and selected['fallback_fraction']==.85
    common = dict(selector='learned',selector_weights=weights,fraction=.5,tile=16,max_age=.5,
                  fallback_fraction=.85,batch_size=32)
    policy = dict(common,execution='masked_dense',cache_training=True) if a.arm=='cache' else {}
    student = make_model(teacher,cfg,saved,**policy)
    for name,param in student.named_parameters():
        param.requires_grad_(name.startswith(('update.','cls_head.','reg_head.','dir_head.')))
    original = {k:v.detach().clone() for k,v in student.state_dict().items()}
    reference = make_model(teacher,cfg,saved,**(dict(common,execution='fixed_sparse') if a.arm=='cache' else {}))
    models = dict(source_dense=make_model(teacher,cfg,saved), trained_dense=make_model(teacher,cfg,saved),
                  trained_cache=make_model(teacher,cfg,saved,**dict(common,execution='fixed_sparse')))
    runtime=json.loads((Path(a.root)/'full_gru/effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(runtime['manifest']).read_text())['train']
    assert not ({r['frame'] for r in train} & {r['frame'] for r in val})
    source_ap=json.loads((Path(a.checkpoint).parent/'report.json').read_text())['metrics']['bev_ap70']
    contract=dict(version='local_cache_aware_v1',arm=a.arm,epochs=a.epochs,tbptt=4,seed=303,
                  source_sha256=a.source_sha256,selector_sha256=a.selector_sha256,data_contract=provenance['contract'],
                  teacher_sha256=provenance['checkpoint']['sha256'],train_scans=len(train),val_scans=len(val),
                  frozen='Pillar, observation E, learned selector, all BN statistics',
                  optimizer='AdamW; projection 1e-4 / U blocks 1e-5 / heads 1e-6; halve each epoch; wd1e-4, clip5',
                  losses=dict(gt=1.,head_kd=.2,feature_kd=.5),fraction=.5,tile=16,max_age=.5,
                  fallback_fraction=.85,deployment='fixed_sparse batch32',
                  training_execution='masked_dense' if a.arm=='cache' else 'dense',
                  state_policy='own recursive state; unselected H retained; four-frame graph then detach, not reset',
                  budgets='3 epochs maximum; 3 hour process alarm; no convergence claim',
                  augmentation=False,artificial_delay=False)
    write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',provenance)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    assert tests.returncode==0
    write_json(out/'status.json',dict(status='RUNNING',phase='engineering'))
    gate=engineering(student,reference,teacher,cfg,train)
    write_json(out/'engineering.json',gate)
    del reference
    # Prove frozen teacher never participates in validation, installing hooks
    # only around validation after all student module copies exist.
    def reject_teacher(module,inputs):
        raise RuntimeError('Teacher inference during validation')
    torch.manual_seed(303); np.random.seed(303)
    optimizer=optimizer_for(student)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    history=[]
    for epoch in range(1,a.epochs+1):
        directory=out/('epoch_%03d'%epoch);directory.mkdir()
        tick=time.monotonic()
        training=train_epoch(a,student,teacher,cfg,train,optimizer,criterion,epoch,directory,out)
        for key,value in student.state_dict().items():
            if key.startswith(('observation.','pillar_vfe.','selector_net.')) or 'running_' in key or 'num_batches_tracked' in key:
                assert torch.equal(value,original[key]),key
        detector={k:v.detach().cpu() for k,v in student.state_dict().items() if not k.startswith('selector_net.')}
        models['trained_dense'].load_state_dict(detector,strict=True)
        cache_weights=dict(detector,**{'selector_net.'+k:v for k,v in weights.items()})
        models['trained_cache'].load_state_dict(cache_weights,strict=True)
        path=directory/'student.pth'
        save_checkpoint(path,dict(model=detector,selector=weights,contract=contract,epoch=epoch))
        write_json(out/'status.json',dict(status='RUNNING',phase='full_validation',epoch=epoch))
        # Real-data equivalence rechecked after weight updates on train only.
        engineering_dir=directory/'engineering';engineering_dir.mkdir()
        paired=[models['trained_dense'],
                make_model(teacher,cfg,dict(model=detector),**dict(common,fraction=1.)),
                models['trained_cache'],
                make_model(teacher,cfg,dict(model=detector),**dict(common,execution='masked_dense'))]
        hooks=[m.register_forward_pre_hook(reject_teacher) for m in
               (teacher.pillar_vfe,teacher.backbone,teacher.cls_head,teacher.reg_head)]
        try:
            check=probe(cfg,paired,train);write_json(engineering_dir/'report.json',check)
            del paired
            report=evaluate_models(cfg,val,models,directory,source_ap)
        finally:
            for hook in hooks:hook.remove()
        history.append(dict(epoch=epoch,training=training,validation=report['results'],
                            checkpoint=str(path),sha256=digest(path),seconds=time.monotonic()-tick))
        write_json(out/'history.json',history)
        for group in optimizer.param_groups:group['lr']*=.5
        save_checkpoint(out/'latest.pth',dict(model=student.state_dict(),optimizer=optimizer.state_dict(),
            contract=contract,epoch=epoch,history=history,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()))
    assert digest(a.checkpoint)==a.source_sha256 and digest(a.selector_checkpoint)==a.selector_sha256
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),history=history,
                                     stop_reason='fixed_budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','output','checkpoint','source-sha256','selector-checkpoint','selector-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=['dense','cache'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.output)
    if not 1<=a.epochs<=3:p.error('Fixed feasibility budget: 1-3 epochs')
    if a.worker:
        def expired(signum,frame):raise TimeoutError('Cache-aware refinement exceeded three hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(10800)
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_cache_aware']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=process.pid,gpu=a.gpu,command=command,created_unix=time.time(),
        timeout_seconds=10800,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid,output=str(out))))


if __name__=='__main__':
    main()

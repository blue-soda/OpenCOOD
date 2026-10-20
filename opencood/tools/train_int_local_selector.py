"""Bounded train-only selector learning on the frozen detector's own cached states."""
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
import torch.nn.functional as F
from opencood.models.sub_modules.int_local_selector import LocalChangeSelector
from opencood.tools.evaluate_int_local_cache import make_model, worker as evaluate_worker
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    setup_args = copy.deepcopy(a)
    for key, value in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None,
                           periodic_last=0, max_age=.25, oracle_stage0=False, oracle_index=0,
                           support_profile=False, zero_uncovered=False).items():
        setattr(setup_args, key, value)
    teacher, cfg, _, provenance = setup(setup_args)
    assert digest(a.checkpoint) == a.source_sha256
    source = torch.load(a.checkpoint, map_location='cpu')
    assert source['contract']['phase'] == 'b' and source['contract']['refinement_variant'] == 'base'
    assert source['contract']['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert source['contract']['data_contract'] == provenance['contract']
    runtime = json.loads((Path(a.root)/'full_gru/effective_runtime.json').read_text())['arguments']
    manifest = json.loads(Path(runtime['manifest']).read_text())
    rows = manifest['train']
    # Separately record frame identities; never use validation features/targets
    # for optimizer updates or early stopping.
    train_ids = {r['frame'] for r in rows}
    val_ids = {r['frame'] for r in manifest['val']}
    assert not (train_ids & val_ids), 'Train/validation manifest overlap'
    net = LocalChangeSelector()
    model = make_model(teacher, cfg, source, selector='learned', selector_weights=net.state_dict(),
                       fraction=a.fraction, tile=a.tile, max_age=a.cache_max_age,
                       execution='masked_dense', fallback_fraction=a.fallback_fraction)
    # make_model freezes everything. Re-enable ONLY the new scorer.
    for param in model.selector_net.parameters():
        param.requires_grad_(True)
    model.collect_selector_targets = True
    model.eval()
    optimizer = torch.optim.AdamW(model.selector_net.parameters(), lr=1e-3, weight_decay=1e-4)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {
        id(p) for p in model.selector_net.parameters()}
    def reject_teacher(module, inputs):
        raise RuntimeError('Original INT teacher is not a selector training input')
    hooks = [m.register_forward_pre_hook(reject_teacher) for m in
             (teacher.pillar_vfe, teacher.backbone, teacher.cls_head, teacher.reg_head)]
    contract = dict(version=LocalChangeSelector.version, source_sha256=a.source_sha256,
                    source_checkpoint=a.checkpoint, data_contract=provenance['contract'],
                    split='train', train_scans=len(rows),
                    validation_scans=len(manifest['val']), disjoint_frame_ids=True,
                    fraction=a.fraction, tile=a.tile, max_age=a.cache_max_age,
                    fallback_fraction=a.fallback_fraction, seed=303, epochs=a.epochs,
                    selector_parameters=sum(p.numel() for p in model.selector_net.parameters()),
                    detector_frozen=True, teacher_execution_forbidden=True,
                    target='log(relative full-U candidate H change from own aligned cached H + 1e-4)',
                    optimizer='AdamW lr=1e-3, halved after each epoch; weight_decay=1e-4; clip_grad_norm=5',
                    loss='smooth_l1 on tile log scores, mean; no GT labels or AP loss',
                    trajectory='learned selector from initialization; full U labels, selected-cell writeback only',
                    deployment_execution=a.execution, batch_size=a.batch_size,
                    budget='fixed epochs <=3 and two-hour process alarm, no extension based on validation',
                    caveat='Scorer-only training, not cache-aware adaptation of detector weights; no speed claim.')
    write_json(out/'training_contract.json', contract)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                            'opencood/test', '-p', 'test_int*.py'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    assert tests.returncode == 0, 'INT tests failed'
    history = []
    started = time.monotonic()
    try:
        for epoch in range(1, a.epochs+1):
            directory = out/('epoch_%03d'%epoch)
            directory.mkdir()
            state = None
            losses, overlaps, writes = [], [], []
            _, stream = loader(rows, cfg, 4, 303+epoch)
            initial = copy.deepcopy(model.selector_net.state_dict())
            for index, sample in enumerate(stream):
                assert sample['meta']['frame'] == rows[index]['frame']
                ego = to_device(sample['ego'], torch.device('cuda'))
                with torch.no_grad():
                    _, state, info = model.step(ego, state, sample['meta'])
                writes.append(info['region_cache']['actual_write_fraction'])
                example = model.selector_example
                if example is not None:
                    logits, target = example
                    loss = F.smooth_l1_loss(logits, target)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.selector_net.parameters(), 5.)
                    assert torch.isfinite(norm)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    k = max(1, int(np.ceil(a.fraction*logits.numel())))
                    left = logits.detach().flatten().topk(k).indices
                    right = target.flatten().topk(k).indices
                    overlaps.append(float((left[:, None] == right[None, :]).any(1).float().mean()))
                    model.selector_example = None
                if (index+1) % 100 == 0:
                    progress = dict(status='RUNNING', phase='train_selector', epoch=epoch,
                                    scans=index+1, total_scans=len(rows),
                                    loss=float(np.mean(losses[-100:])) if losses else None)
                    write_json(out/'status.json', progress)
                    print(json.dumps(progress), flush=True)
            assert losses and any(not torch.equal(initial[k], v) for k,v in model.selector_net.state_dict().items())
            for key, value in model.state_dict().items():
                if not key.startswith('selector_net.'):
                    assert torch.equal(value.cpu(), source['model'][key]), key
            assert all(p.grad is None for name, p in model.named_parameters()
                       if not name.startswith('selector_net.'))
            checkpoint = directory/'selector.pth'
            torch.save(dict(contract, epoch=epoch,
                            selector={k:v.detach().cpu() for k,v in model.selector_net.state_dict().items()},
                            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all()), str(checkpoint))
            record = dict(epoch=epoch, scans=len(rows), optimizer_steps=len(losses),
                          train_log_score_loss=float(np.mean(losses)),
                          train_topk_overlap=float(np.mean(overlaps)),
                          actual_write_fraction=float(np.mean(writes)),
                          detector_unchanged=True, checkpoint=str(checkpoint), sha256=digest(checkpoint))
            write_json(directory/'train.json', record)
            write_json(out/'status.json', dict(status='RUNNING', phase='full_validation', epoch=epoch))
            evaluation = directory/'validation'
            evaluation.mkdir()
            ea = argparse.Namespace(root=a.root, output=str(evaluation), checkpoint=a.checkpoint,
                                    source_sha256=a.source_sha256, gpu=a.gpu, stage='full', frames=256,
                                    fractions=[a.fraction], tile=a.tile, cache_max_age=a.cache_max_age,
                                    fallback_fraction=a.fallback_fraction, execution=a.execution,
                                    batch_size=a.batch_size, selector='learned',
                                    selector_checkpoint=str(checkpoint), selector_sha256=record['sha256'])
            evaluate_worker(ea)
            report = json.loads((evaluation/'report.json').read_text())
            record['validation'] = report['results']
            history.append(record)
            write_json(out/'history.json', history)
            # Fixed protocol; no validation-driven change to scorer or budgets.
            for group in optimizer.param_groups:
                group['lr'] *= .5
        assert digest(a.checkpoint) == a.source_sha256
        write_json(out/'report.json', dict(status='COMPLETED', epochs=len(history), history=history,
                                          seconds=time.monotonic()-started, detector_unchanged=True))
        write_json(out/'status.json', dict(status='COMPLETED', epochs=len(history)))
    finally:
        for hook in hooks:
            hook.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--source-sha256', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--tile', type=int, default=16)
    p.add_argument('--cache-max-age', type=float, default=.5)
    p.add_argument('--fallback-fraction', type=float, default=.85)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--execution', choices=['sparse','fixed_sparse'], default='fixed_sparse')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = Path(a.output)
    if a.fraction not in (.25, .5) or not 1 <= a.epochs <= 3 or a.tile < 1:
        p.error('Bounded scorer pilot requires fraction .25/.5 and 1-3 epochs')
    if not 0 < a.cache_max_age < float('inf') or not 0 < a.fallback_fraction <= 1 or a.batch_size < 1:
        p.error('Invalid regional policy')
    if a.worker:
        def expired(signum, frame):
            raise TimeoutError('Selector training/evaluation exceeded two hours')
        signal.signal(signal.SIGALRM, expired); signal.alarm(7200)
        try:
            worker(a)
        except Exception as exc:
            write_json(out/'status.json', dict(status='FAILED', error=repr(exc))); raise
        finally:
            signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free) < 22000 or int(util) > 5:
        raise RuntimeError('GPU occupied; no shared admission')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'opencood.tools.train_int_local_selector']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=process.pid, gpu=a.gpu, command=command,
        created_unix=time.time(), timeout_seconds=7200,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid, output=str(out))))


if __name__ == '__main__':
    main()

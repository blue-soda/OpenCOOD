"""Audit completed repair training and run three full validations sequentially."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def audit_frames(directory, rows):
    records = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
    assert len(records) == len(rows)
    assert all(a['frame'] == b['frame'] and a['segment'] == b['segment']
               and a['supervised'] == b['supervised'] for a, b in zip(records, rows))
    report = read(directory/'report.json')
    partial = [r for r in records if r['local_partial']]
    assert report['scans'] == len(records) and report['labels'] == sum(r['supervised'] for r in records)
    assert report['partial_frames'] == len(partial)
    assert report['partial_labeled_frames'] == sum(r['supervised'] for r in partial)
    assert all(not r['local_verification_errors'] for r in records), 'No inference teacher allowed'
    assert all(math.isfinite(r['ms']) and r['ms'] > 0 for r in records)
    measured = sum(r['ms'] for r in records[10:])/(len(records)-10)
    assert abs(measured-report['model_step_ms_mean']) < 1e-8
    return dict(scans=len(records), labels=report['labels'], partial_frames=len(partial),
                partial_labeled_frames=report['partial_labeled_frames'])


def worker(a):
    out = Path(a.output)
    source = Path(a.training_root)
    runtime = read(Path(a.root)/'full_gru'/'effective_runtime.json')['arguments']
    rows = read(runtime['manifest'])['val']
    bests, audits = {}, {}
    for policy in ('local', 'dense'):
        run = source/('light_repair_'+policy)
        assert read(run/'status.json')['status'] == 'COMPLETED'
        history, best = read(run/'history.json'), read(run/'best.json')
        assert read(run/'report.json')['stop_reason'] == 'plateau'
        assert len(history) >= 5 and history[-1]['stale_epochs'] >= 5 and history[-1]['lr'] <= .0001
        assert best['ap70'] == max(h['ap70'] for h in history)
        assert digest(best['checkpoint']) == best['sha256']
        epochs = []
        for h in history:
            directory = run/('epoch_%03d'%h['epoch'])
            report = read(directory/'report.json')
            assert report['metrics']['bev_ap70'] == h['ap70']
            assert report['repair_checkpoint_sha256'] == digest(directory/'repairs.pth')
            epochs.append(audit_frames(directory, rows))
        bests[policy] = best
        audits[policy] = dict(best=best, history=history, epochs=epochs)
    write(out/'training_audit.json', dict(status='PASSED', runs=audits,
        caveat='Validates manifests, summaries and hashes; does not independently recompute AP from boxes.'))
    results = {}
    cases = [('control', 'dense', 1, 'none'), ('local', 'local', 2, 'none'), ('dense', 'dense', 2, 'none')]
    if a.suite == 'mechanism':
        cases = [('original', 'dense', 1, 'original'), ('dense', 'dense', 2, 'none'),
                 ('zero_residual', 'dense', 2, 'zero-residual')]
    for name, policy, period, ablation in cases:
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits']).decode().strip())
        assert free > 22000, 'GPU acquired by another job; stop rather than contaminate timings'
        target = out/name
        target.mkdir()
        command = [sys.executable, '-m', 'opencood.tools.run_int_early_cache', '--root', a.root,
                   '--output', str(target), '--gpu', str(a.gpu), '--stage', 'learned-eval',
                   '--source-suite', str(source/'early_four_way_v2'), '--repair-policy', policy,
                   '--repair-checkpoint', bests[policy]['checkpoint'], '--repair-period', str(period),
                   '--repair-ablation', ablation, '--worker']
        write(out/'status.json', dict(status='RUNNING', phase=name))
        with (target/'run.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            write(target/'launch.json', dict(command=command, pid=child.pid, gpu=a.gpu, created_unix=time.time()))
            assert child.wait() == 0, 'Evaluation failed: '+name
        assert read(target/'status.json')['status'] == 'COMPLETED'
        exposure = audit_frames(target, rows)
        report = read(target/'report.json')
        expected = read(Path(a.root)/'full_gru'/'best.json')['bev_ap70'] if period == 1 else bests[policy]['ap70']
        if ablation != 'zero-residual':
            assert abs(report['metrics']['bev_ap70']-expected) < 1e-10, 'Reloaded AP mismatch'
        else:
            assert exposure == results['dense']['audit'], 'Intervention changed cache schedule'
        assert report['repair_checkpoint_sha256'] == bests[policy]['sha256']
        results[name] = dict(report=report, audit=exposure)
        write(out/'results.json', results)
    write(out/'status.json', dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'training-root', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--suite', choices=['standard', 'mechanism'], default='standard')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if a.worker:
        try:
            worker(a)
        except Exception as exc:
            write(out/'status.json', dict(status='FAILED', error=repr(exc)))
            raise
        return
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(Path.cwd()),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'opencood.tools.benchmark_int_light_repair']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    write(out/'launch.json', dict(command=command, pid=child.pid, gpu=a.gpu,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), created_unix=time.time()))
    print(json.dumps(dict(pid=child.pid, output=str(out))))


if __name__ == '__main__':
    main()

"""Adopt the last live snapshot evaluation before its 6h supervisor timeout.

Only the exact own finish coordinator is terminated; its evaluator and workers
stay alive. No predictions, checkpoints, configs or budgets are changed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from opencood.tools.supervise_e3dsnn_snapshot_joint import (
    DRIVER, TRAINER, option, process, same, read, save)

REPO = Path(__file__).resolve().parents[2]
MODULE = 'opencood.tools.supervise_e3dsnn_snapshot_eval'


def validate_commands(parent, child, root, arm):
    folder = root / arm
    expected_parent = {'-m': DRIVER, '--root': str(root), '--arm': arm, '--mode': 'finish'}
    expected_child = {'-m': TRAINER, '--mode': 'eval', '--workers': '2', '--seed': '20260929',
                      '--config': str(folder / 'b8192.yaml'),
                      '--checkpoint': str(folder / 'train/best_ap70.pth'),
                      '--output': str(folder / 'b8192')}
    if any(option(parent, k) != v for k, v in expected_parent.items()):
        raise ValueError('Not the exact finish coordinator')
    if any(option(child, k) != v for k, v in expected_child.items()):
        raise ValueError('Not the exact final 8192-byte evaluator')
    allowed = set(expected_child) | {'--packet-roundtrip', '--eval-3d', '--save-predictions', '--profile'}
    if any(x.startswith('--') and x not in allowed for x in child):
        raise ValueError('Unexpected evaluator override')
    for flag in ('--packet-roundtrip', '--eval-3d', '--save-predictions', '--profile'):
        if child.count(flag) != 1:
            raise ValueError('Missing or repeated required flag ' + flag)


def verify_budget(folder, budget):
    from opencood.tools.launch_e3dsnn_hybrid_round import evaluation, sha
    out = folder / ('b' + str(budget))
    result = evaluation(out)
    packets = [json.loads(s) for s in (out / 'packets/index.jsonl').read_text().splitlines()]
    frames = read(out / 'predictions/index.json')['frames']
    if len(packets) != 1738 or len(frames) != 1738:
        raise ValueError('Incomplete evaluation')
    for packet, frame in zip(packets, frames):
        p = out / 'packets' / packet['file']
        if (sha(p) != packet['sha256'] or p.stat().st_size != packet['actual_bytes']
                or packet['actual_bytes'] != frame['communication']['total_bytes']
                or not 0 <= packet['actual_bytes'] <= budget
                or packet['frame_id'] != frame['frame_id']):
            raise ValueError('Packet integrity, identity or budget mismatch')
    return result


def verify_contract(root, arm):
    from opencood.tools.launch_e3dsnn_hybrid_round import sha
    protocol = read(root / 'protocol.json')
    if sha(root / 'source.pth') != protocol['source_sha256']:
        raise ValueError('Source changed')
    for rel, digest in protocol['config_hashes'].items():
        if sha(root / rel) != digest:
            raise ValueError('Configuration changed')
    folder = root / arm
    if not read(folder / 'audit.json')['passed'] or read(folder / 'train/status.json')['state'] != 'complete':
        raise ValueError('Training/audit not completed')
    return dict(protocol_sha256=sha(root / 'protocol.json'),
                best_sha256=sha(folder / 'train/best_ap70.pth'))


def handoff(root, arm, parent_pid, child_pid):
    folder = root / arm
    parent, child = process(parent_pid), process(child_pid)
    validate_commands(parent['command'], child['command'], root, arm)
    if child['parent'] != parent_pid:
        raise ValueError('Evaluator belongs to another parent')
    status = read(folder / 'status.json')
    if status['state'] != 'running' or status['stage'] != 'b8192':
        raise ValueError('Wrong finalization stage')
    if (folder / 'b8192/evaluation.json').exists():
        raise ValueError('Evaluator already finished; do not race its finalizer')
    hashes = verify_contract(root, arm)
    for budget in (0, 2048, 4096):
        verify_budget(folder, budget)
    # Copy only documented execution controls, never serialize environment secrets.
    entries = (Path('/proc') / str(parent_pid) / 'environ').read_bytes().split(b'\0')
    source_env = {k.decode(): v.decode() for k, v in (x.split(b'=', 1) for x in entries if b'=' in x)}
    env = dict(os.environ)
    for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'OPENCOOD_CUDA_MEMORY_FRACTION', 'PYTHONPATH'):
        env[key] = source_env[key]
    with (folder / 'eval_supervision_handoff.json').open('x') as f:
        json.dump(dict(parent=parent, evaluator=child, contract=hashes, root=str(root), arm=arm,
                       reason='Measured 8KiB evaluation throughput projects beyond 6h timeout',
                       attention_after_hours=36, evaluation_unchanged=True, unix=time.time()), f, indent=2)
    with (folder / 'eval_supervision.log').open('x') as log:
        watcher = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--mode', 'watch',
                                    '--root', str(root), '--arm', arm], cwd=str(REPO), env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    for _ in range(100):
        if (folder / 'eval_supervision_ready.json').exists():
            break
        if watcher.poll() is not None:
            raise RuntimeError('Watcher failed; original supervisor untouched')
        time.sleep(.1)
    else:
        raise TimeoutError('Watcher not ready; original supervisor untouched')
    if not same(parent) or not same(child):
        raise RuntimeError('Process identity changed before handoff')
    os.kill(parent_pid, signal.SIGTERM)
    for _ in range(100):
        if not same(parent):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Original supervisor still alive')
    if not same(child):
        raise RuntimeError('Evaluator did not survive handoff')
    record = dict(passed=True, watcher_pid=watcher.pid, evaluator_pid=child_pid,
                  evaluator_start_ticks=child['start_ticks'], previous_supervisor_pid=parent_pid,
                  evaluator_preserved=True, unix=time.time())
    save(folder / 'eval_supervision_accepted.json', record)
    print(json.dumps(record), flush=True)


def watch(root, arm):
    folder = root / arm
    record = read(folder / 'eval_supervision_handoff.json')
    child = record['evaluator']
    try:
        if not same(child):
            raise RuntimeError('Evaluator missing before handoff')
        save(folder / 'eval_supervision_ready.json', dict(pid=os.getpid(), unix=time.time()))
        for _ in range(600):
            if (folder / 'eval_supervision_accepted.json').exists():
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Handoff not accepted; no finalization')
        started = time.monotonic()
        while same(child):
            if time.monotonic() - started > 36 * 3600:
                save(folder / 'eval_supervision_status.json', dict(state='needs_attention',
                     reason='36h reached; evaluator left running, no retry'))
                return
            save(folder / 'eval_supervision_status.json', dict(state='watching_existing_evaluator',
                 evaluator_pid=child['pid'], elapsed_seconds=time.monotonic()-started, unix=time.time()))
            time.sleep(60)
        if verify_contract(root, arm) != record['contract']:
            raise ValueError('Evaluation provenance changed')
        if not (folder / 'b8192/evaluation.json').exists():
            raise RuntimeError('Evaluator ended without complete result; no automatic retry')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        save(folder / 'status.json', dict(state='running', stage='b8192_replay', unix=time.time()))
        with (folder / 'b8192_replay.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                            '--evaluation-dir', str(folder / 'b8192')], cwd=str(REPO), env=env,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=3600)
        results = {'b'+str(b): verify_budget(folder, b) for b in (0, 2048, 4096, 8192)}
        from opencood.tools.launch_e3dsnn_hybrid_round import sha
        save(folder / 'summary.json', results)
        save(folder / 'status.json', dict(state='complete', sha256=sha(folder / 'summary.json')))
        save(folder / 'eval_supervision_status.json', dict(state='complete', unix=time.time()))
    except Exception:
        error = dict(state='failed', error=traceback.format_exc(), automatic_retry=False)
        save(folder / 'eval_supervision_status.json', error)
        # Pre-handoff failure must not change the still-owned coordinator status.
        if (folder / 'eval_supervision_accepted.json').exists():
            save(folder / 'status.json', error)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=('fixed2k', 'multi248k'), required=True)
    p.add_argument('--mode', choices=('handoff', 'watch'), required=True)
    p.add_argument('--coordinator-pid', type=int)
    p.add_argument('--evaluator-pid', type=int)
    a = p.parse_args()
    if a.mode == 'handoff':
        handoff(a.root, a.arm, a.coordinator_pid, a.evaluator_pid)
    else:
        watch(a.root, a.arm)

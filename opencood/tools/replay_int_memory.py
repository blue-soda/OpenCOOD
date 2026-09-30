"""Recorded, frozen-best full-stream replay and history interventions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from opencood.tools.launch_int_memory import digest


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(temporary), str(path))


def worker(path):
    job = read(path)
    root = Path(path).parent
    status = dict(status='RUNNING', mode=job['mode'], results={})
    try:
        for policy, command in job['commands'].items():
            status['current_policy'] = policy
            write(root/'status.json', status)
            with (root/(policy+'.log')).open('x') as log:
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=7200)
            result = read(root/policy/'result.json')
            summary = result['evaluation']
            if summary['scans'] != job['scans'] or summary['labels'] != job['labels']:
                raise ValueError('Replay coverage mismatch: '+policy)
            provenance = read(root/policy/'manifest.json')
            if provenance['checkpoint_sha256'] != job['checkpoint_sha256']:
                raise ValueError('Replay checkpoint mismatch')
            if provenance['manifest_sha256'] != job['manifest_sha256']:
                raise ValueError('Replay manifest mismatch')
            frames = [json.loads(line) for line in (root/policy/'eval_frames.jsonl').read_text().splitlines()]
            manifest = read(job['manifest'])
            if [r['frame'] for r in frames] != [r['frame'] for r in manifest['val']]:
                raise ValueError('Replay frame order mismatch')
            if any(r.get('invalid') for r in frames):
                raise ValueError('Invalid frame in replay')
            if policy == 'aligned':
                difference = max(abs(summary['metrics'][key]-value)
                                 for key, value in job['reference_metrics'].items())
                if difference > 1e-6:
                    raise ValueError('Frozen-weight replay AP changed: %g' % difference)
                summary['max_ap_replay_difference'] = difference
            status['results'][policy] = summary
            write(root/'status.json', status)
        status.update(status='COMPLETED', completed_unix=time.time())
        write(root/'status.json', status)
    except Exception as exc:
        status.update(status='FAILED', error=type(exc).__name__+': '+str(exc))
        write(root/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--mode', choices=['lif', 'leaky', 'gru'])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.root is None or args.mode is None or args.gpu is None:
        parser.error('Specify root, mode and GPU')
    root = Path(args.root).resolve()
    training = root/('full_'+args.mode)
    completed = read(training/'result.json')
    if completed['status'] not in ('PLATEAU_REACHED', 'EPOCH_LIMIT_REACHED'):
        raise ValueError('Training is still running or failed; do not select a moving best')
    best = read(training/'best.json')
    runtime = read(training/'effective_runtime.json')['arguments']
    contract = read(training/'manifest.json')['contract']
    if digest(best['checkpoint']) != best['sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    if digest(runtime['config']) != contract['config_sha256'] or digest(runtime['manifest']) != contract['manifest_sha256']:
        raise ValueError('Training configuration or data changed')
    memory, utilization = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(memory) < 12000 or int(utilization) > 5:
        raise ValueError('GPU is busy; choose an idle GPU')
    manifest = read(runtime['manifest'])
    curve = read(training/'curve.json')
    selected = next(item for item in curve if item['epoch'] == best['epoch'])
    destination = root/('replay_'+args.mode)
    destination.mkdir(exist_ok=False)
    data_dir = yaml.safe_load(Path(runtime['config']).read_text())['data_dir']
    base = [sys.executable, '-m', 'opencood.tools.run_int_ego', '--eval-only',
            '--mode', args.mode, '--config', runtime['config'], '--data', data_dir,
            '--manifest', runtime['manifest'], '--checkpoint', best['checkpoint'],
            '--seed', str(runtime['seed']), '--workers', str(runtime['workers'])]
    commands = {policy: base+['--history-policy', policy, '--output', str(destination/policy)]
                for policy in ('aligned', 'reset', 'no-align')}
    job = dict(mode=args.mode, gpu=args.gpu, created_unix=time.time(),
               training_status=completed['status'], training_epochs=len(curve), best_epoch=best['epoch'],
               checkpoint=best['checkpoint'], checkpoint_sha256=best['sha256'],
               config_sha256=contract['config_sha256'], manifest_sha256=contract['manifest_sha256'],
               manifest=runtime['manifest'], scans=len(manifest['val']),
               labels=sum(row['supervised'] for row in manifest['val']),
               reference_metrics=selected['evaluation']['metrics'], commands=commands,
               code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    record = destination/'launch.json'
    write(record, job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(Path.cwd()),
               NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (destination/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-m', 'opencood.tools.replay_int_memory',
            '--worker', str(record)], env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(destination/'process.json', dict(pid=process.pid, gpu=args.gpu))
    print(json.dumps(dict(mode=args.mode, pid=process.pid, output=str(destination), best_epoch=best['epoch'])))


if __name__ == '__main__':
    main()

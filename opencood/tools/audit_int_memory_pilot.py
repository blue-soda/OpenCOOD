"""Audit actual pilot coverage, learning, events and exact epoch restart."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from opencood.data_utils.int_sequence_order import segment_order
from opencood.tools.launch_int_memory import digest


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def exact(a, b, path='root'):
    if torch.is_tensor(a):
        require(torch.equal(a, b), 'Restart tensor differs: '+path)
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), 'Restart keys differ: '+path)
        for key in a:
            exact(a[key], b[key], path+'/'+str(key))
    elif isinstance(a, (list, tuple)):
        require(len(a) == len(b), 'Restart length differs: '+path)
        for i, (left, right) in enumerate(zip(a, b)):
            exact(left, right, path+'/'+str(i))
    else:
        require(a == b, 'Restart scalar differs: '+path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--resume-gpu', type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    launch = read(root/'launch_pilot.json')
    manifest = read(root/'pilot_manifest.json')
    checkpoint = launch['jobs'][0]['command']
    checkpoint = checkpoint[checkpoint.index('--checkpoint')+1]
    initial = torch.load(checkpoint, map_location='cpu')
    report = dict(status='RUNNING', parent_manifest_sha256=launch['parent_manifest_sha256'],
                  config_sha256=launch['config_sha256'], checkpoint_sha256=launch['checkpoint_sha256'],
                  pilot_scope='Engineering only; no scientific AP conclusion', modes={})
    for job in launch['jobs']:
        directory, mode = Path(job['output']), job['mode']
        require(not (directory/'failure.json').exists(), 'Pilot has failure evidence: '+mode)
        require(read(directory/'result.json')['status'] == 'EPOCH_LIMIT_REACHED', 'Pilot not completed: '+mode)
        mode_report = {}
        previous = None
        for epoch in (1, 2):
            folder = directory/('epoch_%03d' % epoch)
            train = lines(folder/'train_frames.jsonl')
            evaluation = lines(folder/'eval_frames.jsonl')
            expected = segment_order(manifest['train'], 303+epoch)
            require([r['frame'] for r in train] == [r['frame'] for r in expected], 'Training coverage/order')
            require([r['frame'] for r in evaluation] == [r['frame'] for r in manifest['val']], 'Validation coverage/order')
            summary = read(folder/'train_summary.json')
            require(summary['supervised_frames'] == sum(r['supervised'] for r in expected), 'Supervision coverage')
            gradients = [r['temporal_gradient_norm'] for r in train if r.get('optimizer_step')]
            require(gradients and all(math.isfinite(g) for g in gradients) and max(gradients) > 0, 'Missing FM gradient')
            current = torch.load(str(folder/'trained.pth'), map_location='cpu')['model']
            if epoch == 1:
                for key, value in initial.items():
                    require(torch.equal(current[key], value), 'Frozen spatial parameter changed: '+key)
            else:
                require(any(not torch.equal(current[key], initial[key]) for key in initial), 'Joint phase did not learn spatial weights')
                require(any(not torch.equal(current[key], previous[key]) for key in current if key.startswith('feature_memory.')), 'FM did not learn in joint phase')
            previous = current
            details = dict(scans=len(train), supervised_frames=summary['supervised_frames'],
                           optimizer_steps=summary['updates'], temporal_gradient_max=max(gradients),
                           peak_allocated_bytes=summary['peak_allocated_bytes'])
            if mode in ('lif', 'leaky'):
                rates = [r['emission_nonzero_fraction'] for r in train]
                require(all(math.isfinite(r) for r in rates) and 0 < sum(rates)/len(rates) < 1, 'Degenerate emission')
                details.update(emission_fraction_mean=sum(rates)/len(rates),
                               emission_on_observed_mean=sum(r['emission_on_observed_fraction'] for r in train)/len(train),
                               membrane_abs_max=max(r['membrane_abs_max'] for r in train),
                               trace_abs_max=max(r['trace_abs_max'] for r in train))
            mode_report[str(epoch)] = details
        mode_report['total_parameters'] = sum(value.numel() for key, value in current.items() if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
        report['modes'][mode] = mode_report
        del previous, current
    del initial
    (root/'pilot_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    gpu = str(args.resume_gpu)
    available = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free,utilization.gpu',
                                          '--format=csv,noheader,nounits']).decode().strip().split(',')
    require(int(available[0]) >= 18000 and int(available[1]) <= 5, 'Resume GPU busy')
    command = list(next(job for job in launch['jobs'] if job['mode'] == 'lif')['command'])
    target = root/'pilot_lif_restart'
    require(not target.exists(), 'Restart audit already exists; preserve it and inspect before rerun')
    command[command.index('--output')+1] = str(target)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(Path.cwd()),
               NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'pilot_lif_restart.log').open('x') as log:
        subprocess.run(command+['--stop-after-epoch', '1'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
        require(read(target/'result.json')['status'] == 'CHECKPOINT_BOUNDARY_STOP', 'No boundary checkpoint')
        subprocess.run(command+['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    reference = torch.load(str(root/'pilot_lif/latest.pth'), map_location='cpu')
    resumed = torch.load(str(target/'latest.pth'), map_location='cpu')
    for key in ('model', 'optimizer', 'scheduler', 'contract', 'epoch', 'best', 'significant_best', 'stale'):
        exact(reference[key], resumed[key], key)
    for original_epoch, resumed_epoch in zip(reference['history'], resumed['history']):
        exact(original_epoch['evaluation']['metrics'], resumed_epoch['evaluation']['metrics'], 'AP')
    report.update(status='PASSED', resume='Exact model/optimizer/scheduler/contract/AP equality at epoch 2',
                  gate_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    (root/'engineering_gate.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

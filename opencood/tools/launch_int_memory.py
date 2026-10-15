"""Launch a recorded INT memory comparison on explicitly selected free GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from opencood.tools.prepare_int_pilot import select


def digest(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage', choices=['pilot', 'full'], required=True)
    parser.add_argument('--modes', nargs='+', choices=['lif', 'leaky', 'gru', 'single', 'concat', 'spike_gru'],
                        default=['lif', 'leaky', 'gru'])
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='GPU indices in modes order')
    parser.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/snn/int_ego_memory.yaml')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.modes) or len(args.gpus) != len(args.modes) or len(set(args.modes)) != len(args.modes):
        parser.error('Choose one distinct GPU per distinct mode')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = '' if args.modes == ['lif', 'leaky', 'gru'] else '_'+ '_'.join(args.modes)
    record = root/('launch_'+args.stage+suffix+'.json')
    if record.exists():
        raise ValueError('Launch record already exists; inspect existing processes instead of duplicating')
    if args.stage == 'full':
        gate = json.loads((root/'engineering_gate.json').read_text())
        if gate['status'] != 'PASSED':
            raise ValueError('Engineering gate has not passed')
        if gate['config_sha256'] != digest(args.config) or gate['checkpoint_sha256'] != digest(args.checkpoint):
            raise ValueError('Gate configuration/initialization mismatch')
        if gate['parent_manifest_sha256'] != digest(args.manifest):
            raise ValueError('Gate data mismatch')
        # Additional baselines must exercise their own two-epoch execution path.
        for mode in args.modes:
            if mode not in ('single', 'concat'):
                continue
            pilot = root/('pilot_'+mode)
            result = json.loads((pilot/'result.json').read_text())
            contract = json.loads((pilot/'manifest.json').read_text())['contract']
            if result['status'] != 'EPOCH_LIMIT_REACHED' or result['epochs'] != 2:
                raise ValueError('Baseline pilot incomplete: '+mode)
            pilot_runtime = json.loads((pilot/'effective_runtime.json').read_text())['arguments']
            pilot_manifest = json.loads(Path(pilot_runtime['manifest']).read_text())
            if (pilot_manifest['protocol']['parent_manifest_sha256'] != digest(args.manifest) or
                    contract['config_sha256'] != digest(args.config) or
                    contract['initialization_sha256'] != digest(args.checkpoint) or
                    contract.get('tbptt_steps') != 4 or
                    contract.get('fm_only_epochs') != int(mode != 'single')):
                raise ValueError('Baseline pilot contract mismatch: '+mode)
    report = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits']).decode()
    available = {int(row[0]): (int(row[1]), int(row[2]))
                 for row in (line.split(',') for line in report.strip().splitlines())}
    for gpu in args.gpus:
        if gpu not in available or available[gpu][0] < 18000 or available[gpu][1] > 5:
            raise RuntimeError('GPU %d is not sufficiently idle: %s' % (gpu, available.get(gpu)))
    source = Path(args.manifest).read_bytes()
    manifest = Path(args.manifest).resolve()
    if args.stage == 'pilot':
        original = json.loads(source)
        pilot = {split: select(original[split], 2) for split in ('train', 'val')}
        pilot['protocol'] = dict(purpose='engineering_only',
                                 parent_manifest_sha256=hashlib.sha256(source).hexdigest())
        manifest = root/('pilot_manifest'+suffix+'.json')
        with manifest.open('x') as output:
            json.dump(pilot, output, indent=2)
    command = [sys.executable, '-m', 'opencood.tools.run_int_epochs',
               '--manifest', str(manifest), '--checkpoint', str(Path(args.checkpoint).resolve()),
               '--config', str(Path(args.config).resolve()), '--tbptt-steps', '4',
               '--seed', '303', '--workers', '4', '--lr', '0.0001',
               '--max-epochs', '2' if args.stage == 'pilot' else '40',
               '--min-epochs', '2' if args.stage == 'pilot' else '20',
               '--patience', '10', '--min-delta', '0.001']
    launches = dict(stage=args.stage, created_unix=time.time(),
                    parent_manifest_sha256=hashlib.sha256(source).hexdigest(),
                    config_sha256=digest(args.config), checkpoint_sha256=digest(args.checkpoint),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                    gpu_snapshot=report, jobs=[])
    for mode in args.modes:
        if (root/(args.stage+'_'+mode)).exists():
            raise ValueError('Output exists: '+mode)
    for mode, gpu in zip(args.modes, args.gpus):
        output = root/(args.stage+'_'+mode)
        actual = command + ['--mode', mode, '--output', str(output),
                            '--fm-only-epochs', str(int(mode != 'single'))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(Path.cwd()),
                   NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        log_path = root/(args.stage+'_'+mode+'.log')
        with log_path.open('x') as log:
            process = subprocess.Popen(actual, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, env=env, start_new_session=True)
        launches['jobs'].append(dict(mode=mode, gpu=gpu, pid=process.pid,
                                      command=actual, output=str(output), log=str(log_path)))
        record.write_text(json.dumps(launches, indent=2)+'\n')
    print(json.dumps(launches, indent=2))


if __name__ == '__main__':
    main()

"""Launch the authorized 30-epoch frozen-fusion control after both stage-1 gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(root, initializer_root, activation, gpu):
    root, initializer_root = Path(root).resolve(), Path(initializer_root).resolve()
    repo = Path(__file__).resolve().parents[2]
    summary = json.loads((root / 'stage1_summary.json').read_text())
    assert summary['passed']
    for a in ('count4', 'relu'):
        assert json.loads((root / (a + '_stage1_pipeline.json')).read_text())['state'] == 'complete'
        assert json.loads((root / (a + '_frozen_smoke.json')).read_text())['passed']
    initializer = initializer_root / 'initializers' / ('snn_single.pth' if activation == 'count4' else 'ann_single.pth')
    digest = hashlib.sha256(initializer.read_bytes()).hexdigest()
    assert digest == json.loads((root / (activation + '_frozen_smoke.json')).read_text())['checkpoint_sha256']
    name = activation + '_frozen30'
    lock = root / (name + '_launch.json')
    if lock.exists() or (root / (name + '_pipeline.json')).exists():
        raise FileExistsError('Already attempted; inspect existing run, do not duplicate')
    for other in root.glob('*_launch.json'):
        old = json.loads(other.read_text())
        status = other.with_name(other.name.replace('_launch.json', '_pipeline.json'))
        state = json.loads(status.read_text())['state'] if status.exists() else 'launching'
        if old.get('gpu') == gpu and state not in ('complete', 'failed'):
            raise RuntimeError('GPU reserved by ' + other.name)
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode()
    used, free, util = map(int, row.strip().split(','))
    if used > 1000 or free < 12000 or util > 10:
        raise RuntimeError('Selected GPU is occupied: ' + row)
    py = sys.executable
    config_dir = repo / 'opencood/hypes_yaml/dair-v2x/snn'
    frozen = str(config_dir / ('e3dsnn_pair_' + activation + '_frozen_181.yaml'))
    ego = str(config_dir / ('e3dsnn_pair_' + activation + '_ego_181.yaml'))
    output = root / name
    base = [py, '-u', '-m', 'opencood.tools.train_e3dsnn_single', '--workers', '2']
    jobs = [dict(name=name + '_train', timeout_seconds=43200, command=base +
        ['--mode', 'train', '--config', frozen, '--single-checkpoint', str(initializer),
         '--output', str(output), '--epochs', '30', '--selection-iou', '0.7'])]
    for mode in ('on', 'off'):
        folder = root / (name + '_' + mode)
        command = base + ['--mode', 'eval', '--config', frozen if mode == 'on' else ego,
            '--checkpoint', str(output / 'best_ap70.pth'), '--output', str(folder),
            '--eval-3d', '--save-predictions']
        if mode == 'on':
            command += ['--packet-roundtrip', '--fusion-diagnostics']
        jobs.append(dict(name=name + '_' + mode, timeout_seconds=3600, command=command))
        jobs.append(dict(name=name + '_' + mode + '_replay', timeout_seconds=3600,
            command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                     '--evaluation-dir', str(folder)]))
    jobs.append(dict(name=name + '_diagnostic', timeout_seconds=3600,
        command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.diagnose_e3dsnn_ap70',
                 '--ego-dir', str(root / (name + '_off')), '--fusion-dir', str(root / (name + '_on')),
                 '--output', str(root / (name + '_diagnostic.json'))]))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    plan = dict(name=name, jobs=jobs, gpu=gpu, git_commit=commit, initializer_sha256=digest,
                config_sha256=hashlib.sha256(Path(frozen).read_bytes()).hexdigest())
    plan_path = root / (name + '_plan.json')
    with plan_path.open('x') as stream:
        json.dump(plan, stream, indent=2)
    metadata = dict(state='launching', gpu=gpu, git_commit=commit, started_unix=time.time())
    with lock.open('x') as stream:
        json.dump(metadata, stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(repo),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    with (root / (name + '_supervisor.log')).open('xb') as log:
        process = subprocess.Popen([py, str(root / 'supervise.py'), str(plan_path)],
            cwd=str(repo), env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    metadata.update(state='launched', supervisor_pid=process.pid)
    lock.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--initializer-root', required=True)
    parser.add_argument('--activation', choices=['count4', 'relu'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    launch(args.root, args.initializer_root, args.activation, args.gpu)

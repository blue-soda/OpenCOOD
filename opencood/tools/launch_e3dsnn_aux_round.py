"""Gated lambda=0/1 joint-training experiment after completed frozen controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml


def launch(root, frozen_root, initializer_root, activation, weight, gpu):
    root, frozen_root, initializer_root = map(lambda p: Path(p).resolve(), (root, frozen_root, initializer_root))
    repo = Path(__file__).resolve().parents[2]
    summary_path = frozen_root / 'frozen_final_summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['passed'] and json.loads((root / 'smoke_round.json').read_text())['passed']
    ap70 = summary['models']['count4']['on']['evaluation']['content']['ap']['0.7']
    assert .5503152180393083 < ap70 <= .5798006572452141, 'Predeclared auxiliary branch is not applicable'
    config_dir = repo / 'opencood/hypes_yaml/dair-v2x/snn'
    for a in ('count4', 'relu'):
        assert json.loads((frozen_root / (a + '_frozen30_pipeline.json')).read_text())['state'] == 'complete'
        configs = []
        for w in (0, 1):
            path = config_dir / ('e3dsnn_pair_{}_aux{}_181.yaml'.format(a, w))
            smoke = json.loads((root / ('{}_aux{}_smoke.json'.format(a, w))).read_text())
            assert smoke['passed'] and smoke['config_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            cfg = yaml.safe_load(path.read_text()); cfg.pop('name')
            assert cfg['model']['args'].pop('ego_aux_weight') == w
            configs.append(cfg)
        assert configs[0] == configs[1], 'Matched controls differ beyond auxiliary weight/name'
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3:
        raise RuntimeError('Insufficient reserve disk space')
    stem = '{}_aux{}'.format(activation, weight)
    name = stem + '_joint30'
    lock = root / (name + '_launch.json')
    if lock.exists() or (root / (name + '_pipeline.json')).exists():
        raise FileExistsError('Run already attempted; no duplicate/retry')
    for other in root.glob('*_launch.json'):
        old = json.loads(other.read_text())
        status = other.with_name(other.name.replace('_launch.json', '_pipeline.json'))
        state = json.loads(status.read_text())['state'] if status.exists() else 'launching'
        if old.get('gpu') == gpu and state not in ('complete', 'failed'):
            raise RuntimeError('GPU already reserved: ' + other.name)
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode()
    used, free, util = map(int, row.strip().split(','))
    if used > 1000 or free < 12000 or util > 10:
        raise RuntimeError('GPU is not free: ' + row)
    config = config_dir / ('e3dsnn_pair_' + stem + '_181.yaml')
    ego = config_dir / ('e3dsnn_pair_' + activation + '_ego_181.yaml')
    checkpoint = initializer_root / 'initializers' / ('snn_single.pth' if activation == 'count4' else 'ann_single.pth')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha == json.loads((root / (stem + '_smoke.json')).read_text())['checkpoint_sha256']
    py = sys.executable
    base = [py, '-u', '-m', 'opencood.tools.train_e3dsnn_single', '--workers', '2']
    output = root / name
    jobs = [dict(name=name + '_train', timeout_seconds=43200, command=base +
        ['--mode', 'train', '--config', str(config), '--single-checkpoint', str(checkpoint),
         '--output', str(output), '--epochs', '30', '--warmup-epochs', '3', '--selection-iou', '0.7'])]
    for mode in ('on', 'off'):
        folder = root / (name + '_' + mode)
        command = base + ['--mode', 'eval', '--config', str(config if mode == 'on' else ego),
            '--checkpoint', str(output / 'best_ap70.pth'), '--output', str(folder), '--eval-3d', '--save-predictions']
        if mode == 'on': command += ['--packet-roundtrip', '--fusion-diagnostics']
        jobs.append(dict(name=name + '_' + mode, command=command, timeout_seconds=3600))
        jobs.append(dict(name=name + '_' + mode + '_replay', timeout_seconds=3600,
            command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                     '--evaluation-dir', str(folder)]))
    jobs.append(dict(name=name + '_diagnostic', timeout_seconds=3600,
        command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.diagnose_e3dsnn_ap70',
                 '--ego-dir', str(root / (name + '_off')), '--fusion-dir', str(root / (name + '_on')),
                 '--output', str(root / (name + '_diagnostic.json'))]))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    plan = dict(name=name, jobs=jobs, gpu=gpu, git_commit=commit, initializer_sha256=checkpoint_sha,
                frozen_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    plan_path = root / (name + '_plan.json')
    with plan_path.open('x') as stream: json.dump(plan, stream, indent=2)
    meta = dict(state='launching', gpu=gpu, git_commit=commit, started_unix=time.time())
    with lock.open('x') as stream: json.dump(meta, stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(repo),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    with (root / (name + '_supervisor.log')).open('xb') as log:
        child = subprocess.Popen([py, str(root / 'supervise.py'), str(plan_path)],
            cwd=str(repo), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    meta.update(state='launched', supervisor_pid=child.pid)
    lock.write_text(json.dumps(meta, indent=2)); print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--frozen-root', required=True)
    parser.add_argument('--initializer-root', required=True)
    parser.add_argument('--activation', choices=['count4', 'relu'], required=True)
    parser.add_argument('--weight', type=int, choices=[0, 1], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    launch(args.root, args.frozen_root, args.initializer_root, args.activation, args.weight, args.gpu)

"""One matched receiver-readout trial after the completed auxiliary round."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(root, aux_root, gpu=5):
    root, aux_root = Path(root).resolve(), Path(aux_root).resolve()
    repo = Path(__file__).resolve().parents[2]
    name = 'count4_contaux1_joint30'
    lock = root / (name + '_launch.json')
    if lock.exists() or (root / (name + '_pipeline.json')).exists():
        raise FileExistsError('Run already attempted; no duplicate/retry')
    summary_path = aux_root / 'aux_final_summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['passed'] and json.loads((aux_root / 'cohort_check.json').read_text())['passed']
    control = summary['runs']['count4_aux1_joint30']
    assert control['on']['evaluation']['content']['ap']['0.7'] < .5798006572452141
    assert str(gpu) == control['on']['manifest']['content']['gpu_visible'], 'Use the matched control GPU'
    for stem in summary['runs']:
        assert json.loads((aux_root / (stem + '_pipeline.json')).read_text())['state'] == 'complete'
    config_dir = repo / 'opencood/hypes_yaml/dair-v2x/snn'
    config = config_dir / 'e3dsnn_pair_count4_contaux1_181.yaml'
    original = config_dir / 'e3dsnn_pair_count4_aux1_181.yaml'
    reference = yaml.safe_load((aux_root / 'count4_aux1_joint30/config.yaml').read_text())
    assert yaml.safe_load(original.read_text()) == reference, 'Control config has changed'
    candidate = yaml.safe_load(config.read_text())
    candidate.pop('name'); reference.pop('name')
    assert candidate['model']['args'].pop('fusion_readout') == 'continuous'
    assert candidate == reference, 'Trial differs beyond name/readout'
    smoke = json.loads((root / 'smoke.json').read_text())
    assert smoke['passed'] and smoke['readout_audit']['sender_packets_identical']
    assert smoke['config_sha256'] == sha(config)
    checkpoint = Path('/data1/tzm/workspace/diagnostics/e3dsnn_ap70_20261016_181/initializers/snn_single.pth')
    assert sha(checkpoint) == smoke['checkpoint_sha256'] == '404dfa3aec12b657a7450741974cec7b7af62b1b5e35696dcba37ff42709cdbd'
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3:
        raise RuntimeError('Insufficient reserve disk space')
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits']).decode()
    used, free, util = map(int, row.strip().split(','))
    if used > 1000 or free < 12000 or util > 10:
        raise RuntimeError('GPU is not free: ' + row)
    py = sys.executable
    base = [py, '-u', '-m', 'opencood.tools.train_e3dsnn_single', '--workers', '2']
    output = root / name
    jobs = [dict(name=name + '_train', timeout_seconds=43200, command=base +
        ['--mode', 'train', '--config', str(config), '--single-checkpoint', str(checkpoint),
         '--output', str(output), '--epochs', '30', '--warmup-epochs', '3', '--selection-iou', '0.7'])]
    for mode in ('on', 'off'):
        folder = root / (name + '_' + mode)
        cfg = config if mode == 'on' else config_dir / 'e3dsnn_pair_count4_ego_181.yaml'
        command = base + ['--mode', 'eval', '--config', str(cfg), '--checkpoint', str(output / 'best_ap70.pth'),
                         '--output', str(folder), '--eval-3d', '--save-predictions']
        if mode == 'on': command += ['--packet-roundtrip', '--fusion-diagnostics']
        jobs.append(dict(name=name + '_' + mode, timeout_seconds=3600, command=command))
        jobs.append(dict(name=name + '_' + mode + '_replay', timeout_seconds=3600,
            command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                     '--evaluation-dir', str(folder)]))
    jobs.append(dict(name=name + '_diagnostic', timeout_seconds=3600,
        command=['env', 'CUDA_VISIBLE_DEVICES=', py, '-u', '-m', 'opencood.tools.diagnose_e3dsnn_ap70',
                 '--ego-dir', str(root / (name + '_off')), '--fusion-dir', str(root / (name + '_on')),
                 '--output', str(root / (name + '_diagnostic.json'))]))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    plan = dict(name=name, jobs=jobs, gpu=gpu, git_commit=commit, config_sha256=sha(config),
                initializer_sha256=sha(checkpoint), control_summary_sha256=sha(summary_path))
    plan_path = root / (name + '_plan.json')
    with plan_path.open('x') as stream: json.dump(plan, stream, indent=2)
    meta = dict(state='launching', gpu=gpu, git_commit=commit, started_unix=time.time())
    with lock.open('x') as stream: json.dump(meta, stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(repo), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    with (root / (name + '_supervisor.log')).open('xb') as log:
        child = subprocess.Popen([py, str(root / 'supervise.py'), str(plan_path)], cwd=str(repo), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    meta.update(state='launched', supervisor_pid=child.pid)
    lock.write_text(json.dumps(meta, indent=2)); print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--aux-root', required=True)
    parser.add_argument('--gpu', type=int, default=5)
    args = parser.parse_args()
    launch(args.root, args.aux_root, args.gpu)

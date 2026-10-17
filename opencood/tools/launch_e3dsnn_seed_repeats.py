"""Two predeclared SNN fine-tuning repetitions; report all seeds, never best seed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SEEDS = (20260930, 20261001)


def repeated_jobs(plan, old_root, new_root, seed):
    if seed not in SEEDS:
        raise ValueError('Only the two predeclared repeat seeds are allowed')
    old_name = 'count4_aux1_joint30'
    name = 'count4_aux1_seed{}_joint30'.format(seed)
    assert plan['name'] == old_name
    jobs = []
    for old in plan['jobs']:
        job = dict(old)
        job['name'] = old['name'].replace(old_name, name)
        job['command'] = [x.replace(str(old_root) + '/', str(new_root) + '/').replace(old_name, name)
                          for x in old['command']]
        if 'opencood.tools.train_e3dsnn_single' in job['command']:
            assert '--seed' not in job['command']
            job['command'] += ['--seed', str(seed)]
        jobs.append(job)
    return name, jobs


def launch(root, aux_root, continuous_root, seed, gpu):
    root, aux_root, continuous_root = [Path(p).resolve() for p in (root, aux_root, continuous_root)]
    repo = Path(__file__).resolve().parents[2]
    source = aux_root / 'count4_aux1_joint30_plan.json'
    control = json.loads(source.read_text())
    name, jobs = repeated_jobs(control, aux_root, root, seed)
    lock = root / (name + '_launch.json')
    if lock.exists() or (root / (name + '_pipeline.json')).exists():
        raise FileExistsError('Repeat already attempted; no duplicate or automatic retry')
    aux = json.loads((aux_root / 'aux_final_summary.json').read_text())
    continuous = json.loads((continuous_root / 'continuous_final_summary.json').read_text())
    assert aux['passed'] and continuous['passed']
    discrete_ap = aux['runs']['count4_aux1_joint30']['on']['evaluation']['content']['ap']['0.7']
    continuous_ap = continuous['models']['count4']['on']['evaluation']['content']['ap']['0.7']
    assert continuous_ap < discrete_ap < .5798006572452141
    for folder, stem in [(aux_root, 'count4_aux1_joint30'), (continuous_root, 'count4_contaux1_joint30')]:
        assert json.loads((folder / (stem + '_pipeline.json')).read_text())['state'] == 'complete'
    train = jobs[0]['command']
    config = Path(train[train.index('--config') + 1])
    checkpoint = Path(train[train.index('--single-checkpoint') + 1])
    assert hashlib.sha256(config.read_bytes()).hexdigest() == control['config_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == control['initializer_sha256']
    assert train[train.index('--epochs') + 1] == '30' and train[train.index('--warmup-epochs') + 1] == '3'
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3:
        raise RuntimeError('Insufficient disk reserve')
    for other in root.glob('*_launch.json'):
        meta = json.loads(other.read_text())
        state_path = other.with_name(other.name.replace('_launch.json', '_pipeline.json'))
        state = json.loads(state_path.read_text())['state'] if state_path.exists() else 'launching'
        if meta['gpu'] == gpu and state not in ('complete', 'failed'):
            raise RuntimeError('GPU reserved by another repeat')
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode()
    used, free, util = map(int, row.strip().split(','))
    if used > 1000 or free < 12000 or util > 10:
        raise RuntimeError('GPU is not free: ' + row)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    plan = dict(name=name, jobs=jobs, seed=seed, gpu=gpu, git_commit=commit,
                config_sha256=control['config_sha256'], initializer_sha256=control['initializer_sha256'],
                source_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                report_policy='Report all three seeds (20260929,20260930,20261001) and mean/sample SD; no best-seed selection. '
                              'Fixed original single-agent initializer; only fine-tuning randomness is repeated.')
    plan_path = root / (name + '_plan.json')
    with plan_path.open('x') as stream: json.dump(plan, stream, indent=2)
    meta = dict(state='launching', seed=seed, gpu=gpu, git_commit=commit, started_unix=time.time())
    with lock.open('x') as stream: json.dump(meta, stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(repo), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    with (root / (name + '_supervisor.log')).open('xb') as log:
        child = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan_path)], cwd=str(repo),
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    meta.update(state='launched', supervisor_pid=child.pid)
    lock.write_text(json.dumps(meta, indent=2)); print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--aux-root', required=True)
    parser.add_argument('--continuous-root', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    launch(args.root, args.aux_root, args.continuous_root, args.seed, args.gpu)

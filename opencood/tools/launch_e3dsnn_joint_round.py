"""Audited joint prototype: full zero gate, ten epochs, real-wire final replay.

One suitable GPU is sufficient; explicitly authorized shared use is supported.
Mechanism ablations follow this first executable
joint candidate; no old Residual repetition is automatically launched here.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu
from opencood.tools.launch_e3dsnn_content_round import make_jobs, summarize
from opencood.tools.train_e3dsnn_single import write_json


def shared_admission(samples, fraction):
    """Use several samples, not a momentary idle reading; reserve outside Torch."""
    if not math.isfinite(fraction) or not 0 < fraction <= .5:
        raise ValueError('Shared allocator fraction must be in (0, .5]')
    if len(samples) < 3:
        raise ValueError('At least three utilization samples required')
    total = samples[0]['total_mib']
    if any(s['total_mib'] != total for s in samples):
        raise ValueError('GPU capacity changed')
    required = max(8192, math.ceil(total * fraction) + 2048)
    return dict(accepted=min(s['free_mib'] for s in samples) >= required and
                sum(s['util_percent'] for s in samples) / len(samples) <= 50,
                minimum_free_mib=required, allocator_fraction=fraction,
                mean_util_percent=sum(s['util_percent'] for s in samples) / len(samples),
                samples=samples)


def check_gpu(gpu, allow_shared=False, fraction=.2):
    if not allow_shared:
        free_gpu(gpu)
        return dict(accepted=True, policy='exclusive')
    samples = []
    for i in range(5):
        row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits']).decode().strip()
        total, free, util = map(int, row.split(','))
        samples.append(dict(total_mib=total, free_mib=free, util_percent=util))
        if i < 4: time.sleep(2)
    result = shared_admission(samples, fraction)
    result['policy'] = 'user_authorized_shared'
    if not result['accepted']:
        raise RuntimeError('GPU {} shared admission failed: {}'.format(gpu, result))
    return result


def seeded_jobs(root, stage, training_seed):
    """Change training RNG only; keep the registered evaluation seed fixed."""
    if type(training_seed) is not int or not 0 <= training_seed < 2**32:
        raise ValueError('Training seed must fit NumPy uint32')
    name, jobs = make_jobs(root, 'joint', stage)
    changed = 0
    for job in jobs:
        command = job['command']
        if '--mode' in command and command[command.index('--mode') + 1] == 'train':
            command[command.index('--seed') + 1] = str(training_seed)
            changed += 1
    if changed != (1 if stage == 'adapt10' else 0):
        raise ValueError('Unexpected number of training jobs')
    return name, jobs


def prepare(root, audit_root):
    entry = read(audit_root / 'audit.json')
    if not (entry['passed'] and entry['joint_membrane'] and
            entry['local_membrane_perturbation_response_max'] > 0):
        raise ValueError('Real joint detector audit must pass first')
    for filename, key in (('initial.pth', 'initial_sha256'), ('config.yaml', 'config_sha256')):
        assert sha(audit_root / filename) == entry[key]
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()): raise FileExistsError('Fresh joint round required')
    folder = root / 'joint'; folder.mkdir()
    shutil.copyfile(str(audit_root / 'initial.pth'), str(folder / 'initial.pth'))
    raw = yaml.safe_load((audit_root / 'config.yaml').read_text())
    assert raw['model']['core_method'] == 'e3dsnn_joint_detector'
    assert raw['train_params'] == dict(batch_size=2, epoches=10)
    assert raw['optimizer']['lr'] == 5e-5
    entry = copy.deepcopy(entry); entry['config_hashes'] = {}
    for mode in ('p1', 'p2', 'p3', 'off', 'full'):
        cfg = copy.deepcopy(raw)
        if mode.startswith('p'): cfg['model']['args']['readout_prefix'] = int(mode[1:])
        elif mode == 'off': cfg['model']['args']['fusion_mode'] = 'ego'
        else: cfg['model']['args']['content_full_packet'] = True
        path = folder / (mode + '.yaml'); path.write_text(yaml.safe_dump(cfg))
        entry['config_hashes'][mode] = sha(path)
    supervisor = root.parent / 'e3dsnn_budget_round1_20261018/supervise.py'
    shutil.copyfile(str(supervisor), str(root / 'supervise.py'))
    write_json(root / 'audit.json', dict(passed=True, groups=dict(joint=entry),
        real_audit_path=str(audit_root), real_audit_sha256=sha(audit_root / 'audit.json')))
    write_json(root / 'round_status.json', dict(state='prepared', training_started=False))


def coordinate(root):
    try:
        setup = read(root / 'launch.json'); gpu = setup['gpu']
        for stage in ('zero', 'adapt10'):
            if stage == 'adapt10': assert read(root / 'zero_summary.json')['passed']
            # Other users may become busy between stages. Wait without changing
            # experiment inputs or restarting completed work; never kill peers.
            while True:
                try:
                    admission = check_gpu(gpu, setup.get('allow_shared', False),
                                          setup.get('memory_fraction', .2))
                    break
                except RuntimeError as error:
                    write_json(root / 'round_status.json', dict(state='waiting_resources',
                        stage=stage, gpu=gpu, note=str(error), training_started=False))
                    time.sleep(60)
            write_json(root / (stage + '_gpu_admission.json'), admission)
            name, jobs = seeded_jobs(root, stage, setup.get('training_seed', 20260929))
            plan = root / (name + '_plan.json')
            write_json(plan, dict(name=name, jobs=jobs, gpu=gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            if setup.get('allow_shared', False):
                env['OPENCOOD_CUDA_MEMORY_FRACTION'] = str(setup['memory_fraction'])
            with (root / (name + '_supervisor.log')).open('xb') as log:
                process = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan)],
                    cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root / 'round_status.json', dict(state='running', stage=stage,
                supervisor_pid=process.pid, training_started=stage == 'adapt10'))
            if process.wait() != 0: raise RuntimeError('Joint pipeline failed')
            summarize(root, stage)
            if stage == 'adapt10':
                cfg = yaml.safe_load((root / 'joint/p3.yaml').read_text())
                if not cfg['model']['args'].get('joint_learn_order', True):
                    report = read(root / 'adapt10_summary.json')
                    if report['groups']['joint']['selected_sender_weights_changed']:
                        report['passed'] = False
                        write_json(root / 'adapt10_summary.json', report)
                        raise RuntimeError('Frozen sender changed during ablation')
        write_json(root / 'round_status.json', dict(state='complete', finished_unix=time.time()))
    except Exception:
        write_json(root / 'round_status.json', dict(state='failed', error=traceback.format_exc(),
                   note='No automatic retry'))
        raise


def launch(root, gpu, allow_shared=False, memory_fraction=.2, training_seed=20260929):
    if type(training_seed) is not int or not 0 <= training_seed < 2**32:
        raise ValueError('Training seed must fit NumPy uint32')
    audit = read(root / 'audit.json'); assert audit['passed']
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    entry = audit['groups']['joint']
    assert sha(root / 'joint/initial.pth') == entry['initial_sha256']
    for mode, digest in entry['config_hashes'].items():
        assert sha(root / 'joint' / (mode + '.yaml')) == digest
    if shutil.disk_usage(str(root)).free < 6 * 1024**3: raise RuntimeError('Insufficient reserve')
    admission = check_gpu(gpu, allow_shared, memory_fraction)
    setup = dict(gpu=gpu, audit_sha256=sha(root / 'audit.json'), started_unix=time.time(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        training_cap_seconds=21600, per_evaluation_cap_seconds=3600,
        training_seed=training_seed, evaluation_seed=20260929,
        allow_shared=allow_shared, memory_fraction=memory_fraction if allow_shared else None,
        gpu_admission=admission)
    write_json(root / 'launch.json', setup)
    with (root / 'coordinator.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_joint_round',
            '--root', str(root), '--coordinate'], cwd=str(REPO), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(dict(coordinator_pid=process.pid, **setup), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--audit-root', default='/data1/tzm/workspace/diagnostics/e3dsnn_joint_audit_v2_20261019')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--allow-shared', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=.2)
    parser.add_argument('--training-seed', type=int, default=20260929)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--coordinate', action='store_true')
    args = parser.parse_args(); root = Path(args.root)
    if args.prepare: prepare(root, Path(args.audit_root))
    elif args.coordinate: coordinate(root)
    else: launch(root, args.gpu, args.allow_shared, args.memory_fraction, args.training_seed)

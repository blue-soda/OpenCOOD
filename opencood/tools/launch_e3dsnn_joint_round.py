"""Audited joint prototype: full zero gate, ten epochs, real-wire final replay.

One free GPU is sufficient. Mechanism ablations follow this first executable
joint candidate; no old Residual repetition is automatically launched here.
"""
import argparse
import copy
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
            free_gpu(gpu)
            name, jobs = make_jobs(root, 'joint', stage)
            plan = root / (name + '_plan.json')
            write_json(plan, dict(name=name, jobs=jobs, gpu=gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (root / (name + '_supervisor.log')).open('xb') as log:
                process = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan)],
                    cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root / 'round_status.json', dict(state='running', stage=stage,
                supervisor_pid=process.pid, training_started=stage == 'adapt10'))
            if process.wait() != 0: raise RuntimeError('Joint pipeline failed')
            summarize(root, stage)
        write_json(root / 'round_status.json', dict(state='complete', finished_unix=time.time()))
    except Exception:
        write_json(root / 'round_status.json', dict(state='failed', error=traceback.format_exc(),
                   note='No automatic retry'))
        raise


def launch(root, gpu):
    audit = read(root / 'audit.json'); assert audit['passed']
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    entry = audit['groups']['joint']
    assert sha(root / 'joint/initial.pth') == entry['initial_sha256']
    for mode, digest in entry['config_hashes'].items():
        assert sha(root / 'joint' / (mode + '.yaml')) == digest
    if shutil.disk_usage(str(root)).free < 6 * 1024**3: raise RuntimeError('Insufficient reserve')
    free_gpu(gpu)
    setup = dict(gpu=gpu, audit_sha256=sha(root / 'audit.json'), started_unix=time.time(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        training_cap_seconds=21600, per_evaluation_cap_seconds=3600)
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
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--coordinate', action='store_true')
    args = parser.parse_args(); root = Path(args.root)
    if args.prepare: prepare(root, Path(args.audit_root))
    elif args.coordinate: coordinate(root)
    else: launch(root, args.gpu)

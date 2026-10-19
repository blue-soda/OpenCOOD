"""Repeat the learned-content candidate with two new adaptation seeds per family.

Reuse audited, byte-identical initial weights/configs; never reuse a trained best
as initialization. Final evaluation RNG and third-prefix checkpoint selection
remain unchanged. No automatic retry or hyperparameter search.
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

from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu
from opencood.tools.launch_e3dsnn_content_round import make_jobs, summarize
from opencood.tools.train_e3dsnn_single import write_json

SEEDS = (20260930, 20260931)
GROUPS = tuple((family + '_seed' + str(seed), family + '_learn', seed)
               for family in ('snn', 'ann') for seed in SEEDS)
SOURCE_FINAL_SHA = 'a182257ab9d88c60c4dbe238dfb9e9cd8b198817b6266f1426e5741bd65771f3'
SOURCE_ZERO_SHA = 'f1601166330df42e8031a2925bc6b3cd3ede9fce498a7e8c7700771ad94b5298'


def prepare(root, source):
    assert read(source / 'round_status.json')['state'] == 'complete'
    for name, digest in (('adapt10_summary.json', SOURCE_FINAL_SHA),
                         ('zero_summary.json', SOURCE_ZERO_SHA)):
        assert sha(source / name) == digest and read(source / name)['passed']
    original = read(source / 'audit.json')
    assert original['passed']
    # Audit reuse is valid only while the relevant implementation is unchanged.
    source_commit = read(source / 'launch.json')['code_commit']
    paths = ['opencood/models/e3dsnn*', 'opencood/models/fuse_modules/e3dsnn*',
             'opencood/utils/*packet*', 'opencood/tools/train_e3dsnn_single.py',
             'opencood/tools/audit_e3dsnn_content_detector.py']
    for revisions in ([source_commit, 'HEAD'], []):
        subprocess.run(['git', 'diff', '--exit-code'] + revisions + ['--'] + paths,
                       cwd=str(REPO), check=True)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Fresh repetition directory required')
    report = dict(passed=False, groups={}, inherited_interface_audit=True,
                  source_root=str(source), source_audit_sha256=sha(source / 'audit.json'),
                  source_zero_sha256=SOURCE_ZERO_SHA, source_final_sha256=SOURCE_FINAL_SHA,
                  source_code_commit=source_commit, evaluation_seed=20260929)
    for group, original_group, seed in GROUPS:
        entry = copy.deepcopy(original['groups'][original_group])
        assert entry['learn_content']
        entry.update(training_seed=seed, inherited_from=original_group,
                     audit_reuse='Prior interface/zero audit; not a new GPU audit')
        folder = root / group
        folder.mkdir()
        expected = dict(entry['config_hashes'])
        expected['initial'] = entry['initial_sha256']
        for name, digest in expected.items():
            filename = 'initial.pth' if name == 'initial' else name + '.yaml'
            assert sha(source / original_group / filename) == digest
            shutil.copyfile(str(source / original_group / filename), str(folder / filename))
            assert sha(folder / filename) == digest
        report['groups'][group] = entry
    shutil.copyfile(str(source / 'supervise.py'), str(root / 'supervise.py'))
    report['passed'] = True
    write_json(root / 'audit.json', report)
    write_json(root / 'round_status.json', dict(state='prepared', note='Not launched; requires four free GPUs'))


def repeat_jobs(root, group, seed):
    name, jobs = make_jobs(root, group, 'adapt10')
    train = jobs[0]['command']
    assert train[train.index('--mode') + 1] == 'train'
    train[train.index('--seed') + 1] = str(seed)
    for job in jobs[1:]:
        command = job['command']
        if '--seed' in command:
            assert command[command.index('--seed') + 1] == '20260929'
    return name, jobs


def coordinate(root):
    try:
        setup = read(root / 'launch.json')
        for gpu in setup['gpus']:
            free_gpu(gpu)
        processes = []
        for (group, _, seed), gpu in zip(GROUPS, setup['gpus']):
            name, jobs = repeat_jobs(root, group, seed)
            plan = root / (name + '_plan.json')
            write_json(plan, dict(name=name, jobs=jobs, gpu=gpu))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (root / (name + '_supervisor.log')).open('xb') as log:
                process = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan)],
                    cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((name, process))
        write_json(root / 'round_status.json', dict(state='running', stage='adapt10',
                   supervisors={name: process.pid for name, process in processes}))
        while any(p.poll() is None for _, p in processes):
            if any(p.poll() not in (None, 0) for _, p in processes):
                raise RuntimeError('Repeat pipeline failed')
            time.sleep(15)
        if any(p.returncode for _, p in processes):
            raise RuntimeError('Repeat pipeline failed')
        summarize(root, 'adapt10')
        write_json(root / 'round_status.json', dict(state='complete', finished_unix=time.time()))
    except Exception:
        write_json(root / 'round_status.json', dict(state='failed', error=traceback.format_exc(),
                   note='No automatic retry; other jobs not killed'))
        raise


def launch(root, gpus):
    audit = read(root / 'audit.json')
    assert audit['passed'] and audit['inherited_interface_audit']
    if (root / 'launch.json').exists():
        raise FileExistsError('Already attempted')
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError('Four distinct free GPUs required')
    for group, _, seed in GROUPS:
        entry = audit['groups'][group]
        assert entry['training_seed'] == seed
        assert sha(root / group / 'initial.pth') == entry['initial_sha256']
        for name, digest in entry['config_hashes'].items():
            assert sha(root / group / (name + '.yaml')) == digest
    if shutil.disk_usage(str(root)).free < 12 * 1024 ** 3:
        raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus:
        free_gpu(gpu)
    setup = dict(gpus=gpus, code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        audit_sha256=sha(root / 'audit.json'), started_unix=time.time(), training_cap_seconds=21600)
    write_json(root / 'launch.json', setup)
    with (root / 'coordinator.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m',
            'opencood.tools.launch_e3dsnn_content_repeat', '--root', str(root), '--coordinate'],
            cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(dict(coordinator_pid=process.pid, **setup), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', default='/data1/tzm/workspace/diagnostics/e3dsnn_content_round1_v2_20261019')
    parser.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 5])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--coordinate', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    if args.prepare:
        prepare(root, Path(args.source))
    elif args.coordinate:
        coordinate(root)
    else:
        launch(root, args.gpus)

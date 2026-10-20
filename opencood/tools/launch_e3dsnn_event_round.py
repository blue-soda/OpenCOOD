"""Matched sender-only IF-stream adaptation and automatic full wire replay.

Selection is pre-registered at 4096 bytes, including epoch zero. Both groups
use the same source, seed, optimizer and frozen receiver. No automatic retry.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, evaluation
from opencood.tools.launch_e3dsnn_joint_round import check_gpu
from opencood.tools.train_e3dsnn_single import write_json

SOURCE_SHA = 'f4a5795b8b001a7cc4c04fd9eb3959f02df70181afe23691e733cb3aa4eb8588'
BUDGETS = (0, 2048, 4096, 6144, 8192, 16384)
GROUPS = {'fixed4096': [4096, 4096], 'log2048_16384': [2048, 16384]}
RECEIVER_GROUPS = ('decode', 'receiver')


def prepare(root, audit_root):
    audit = read(audit_root / 'audit.json')
    assert audit['passed'] and audit['all_other_parameters_and_buffers_unchanged']
    assert audit['transport'] == 'session_zlib' and audit['sampling'] == 'log_uniform'
    assert audit['trainable'] == ['content.project.weight']
    source = Path(audit['source']); assert sha(source) == audit['source_sha256'] == SOURCE_SHA
    raw = yaml.safe_load((audit_root / 'config.yaml').read_text())
    assert raw['optimizer']['lr'] == 5e-5
    assert raw['train_params'] == dict(batch_size=2, epoches=10)
    assert raw['model']['core_method'] == 'e3dsnn_event_detector'
    assert raw['model']['args']['event_stream']['transport'] == 'session_zlib'
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(str(source), str(root / 'initial.pth'))
    hashes = {}
    for group, bounds in GROUPS.items():
        folder = root / group; folder.mkdir()
        cfg = copy.deepcopy(raw)
        cfg['model']['args']['event_stream'].update(train_budget_range=bounds, sampling='log_uniform', budget_bytes=4096)
        for mode, budget in [('train', 4096)] + [('b' + str(b), b) for b in BUDGETS]:
            current = copy.deepcopy(cfg)
            current['model']['args']['event_stream']['budget_bytes'] = budget
            path = folder / (mode + '.yaml')
            path.write_text(yaml.safe_dump(current))
            hashes[str(path.relative_to(root))] = sha(path)
    write_json(root / 'protocol.json', dict(source_sha256=SOURCE_SHA,
        audit_path=str(audit_root / 'audit.json'), audit_sha256=sha(audit_root / 'audit.json'),
        config_hashes=hashes, groups=GROUPS, training_seed=20260929,
        selection='max validation BEV AP70 at 4096 bytes over epochs 0..10',
        selection_limitation='4096-byte anchor may favor fixed-budget training; report full curve',
        final_budgets_bytes=list(BUDGETS), trainable=['content.project.weight'],
        training_cap_seconds=43200, evaluation_cap_seconds=7200))


def check_frozen(root, folder):
    initial = torch.load(str(root / 'initial.pth'), map_location='cpu')
    protocol = read(root / 'protocol.json')
    allowed = set(protocol.get('trainable_by_group', {}).get(folder.name, ['content.project.weight']))
    report = {}
    for filename in ('best_ap70.pth', 'last.pt'):
        state = torch.load(str(folder / 'train' / filename), map_location='cpu')
        if filename == 'last.pt':
            assert state['epoch'] == 10
            state = state['model']
        assert set(state) == set(initial)
        changed = [k for k in state if not torch.equal(state[k], initial[k])]
        assert set(changed) <= allowed, changed
        if filename == 'last.pt': assert changed
        report[filename] = dict(sha256=sha(folder / 'train' / filename), changed=changed)
    write_json(folder / 'all_frozen_verified.json', dict(passed=True, checkpoints=report))
    return report


def prepare_receivers(root, audit_root):
    """One bounded receiver-mapping study, with the same frozen E1 sender."""
    source_sha = '17b793cd47595570f2be37a07cd3977c825058c2ac704508637927c41717a27d'
    entries = {g: read(audit_root / g / 'audit.json') for g in RECEIVER_GROUPS}
    for group, audit in entries.items():
        assert audit['passed'] and audit['all_other_parameters_and_buffers_unchanged']
        assert audit['cached_source_event_count_and_bytes_unchanged']
        assert audit['train_scope'] == group and audit['source_sha256'] == source_sha
        assert sha(Path(audit['source'])) == source_sha
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(entries['decode']['source'], str(root / 'initial.pth'))
    hashes = {}; trainable = {}
    for group in RECEIVER_GROUPS:
        folder = root / group; folder.mkdir()
        cfg = yaml.safe_load((audit_root / group / 'config.yaml').read_text())
        assert cfg['model']['args']['event_stream']['train_scope'] == group
        assert cfg['model']['args']['event_stream']['train_budget_range'] == [2048,16384]
        assert cfg['optimizer']['lr'] == 5e-5 and cfg['train_params'] == dict(batch_size=2,epoches=10)
        trainable[group] = entries[group]['trainable']
        for mode,budget in [('train',4096)]+[('b'+str(b),b) for b in BUDGETS]:
            current=copy.deepcopy(cfg); current['model']['args']['event_stream']['budget_bytes']=budget
            path=folder/(mode+'.yaml'); path.write_text(yaml.safe_dump(current))
            hashes[str(path.relative_to(root))]=sha(path)
    audits={str(audit_root/g/'audit.json'):sha(audit_root/g/'audit.json') for g in RECEIVER_GROUPS}
    write_json(root/'protocol.json',dict(source_sha256=source_sha,
        audit_path=str(audit_root/'decode/audit.json'),audit_sha256=sha(audit_root/'decode/audit.json'),
        audits=audits,config_hashes=hashes,groups={g:[2048,16384] for g in RECEIVER_GROUPS},
        trainable_by_group=trainable,training_seed=20260929,
        selection='max validation BEV AP70 at 4096 bytes over epochs 0..10',
        final_budgets_bytes=list(BUDGETS),training_cap_seconds=43200,evaluation_cap_seconds=7200,
        hypothesis='Receiver mapping trained for voxel counts limits the new frozen IF sender',
        control='same initial E1 random-budget checkpoint; decode only versus decode plus existing gate/scale',
        limitations='additional adaptation; not a matched historical SNN/ANN architecture comparison',
        exit='one 10-epoch round only; no spatial block or parameter sweep if communication/AP remains poor'))


def run_group(root, group):
    folder = root / group
    status = folder / 'pipeline_status.json'
    protocol = read(root / 'protocol.json')
    launch = read(root / 'launch.json')
    try:
        assert sha(root / 'initial.pth') == protocol['source_sha256']
        for name, digest in protocol['config_hashes'].items():
            if name.startswith(group + '/'): assert sha(root / name) == digest
        base = [sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single',
                '--workers', '2', '--seed', '20260929']
        jobs = [('train', base + ['--mode', 'train', '--config', str(folder / 'train.yaml'),
            '--checkpoint', str(root / 'initial.pth'), '--output', str(folder / 'train'),
            '--epochs', '10', '--selection-iou', '0.7', '--retain-initial-best'],
            protocol['training_cap_seconds'])]
        for budget in BUDGETS:
            name = 'b' + str(budget); output = folder / name
            jobs.append((name, base + ['--mode', 'eval', '--config', str(folder / (name + '.yaml')),
                '--checkpoint', str(folder / 'train/best_ap70.pth'), '--output', str(output),
                '--eval-3d', '--save-predictions', '--profile', '--packet-roundtrip'],
                protocol['evaluation_cap_seconds']))
            jobs.append((name + '_replay', [sys.executable, '-u', '-m',
                'opencood.tools.replay_e3dsnn_predictions', '--evaluation-dir', str(output)],
                protocol['evaluation_cap_seconds']))
        write_json(folder / 'plan.json', dict(jobs=jobs))
        for name, command, timeout in jobs:
            env = dict(os.environ)
            if name.endswith('_replay'): env['CUDA_VISIBLE_DEVICES'] = ''
            elif name != 'train':
                # Recheck availability after long stages, without disrupting peers.
                admission = check_gpu(launch['gpus'][group], True, .2)
                write_json(folder / (name + '_gpu_admission.json'), admission)
            write_json(status, dict(state='running', stage=name, started_unix=time.time()))
            with (folder / (name + '.log')).open('xb') as log:
                subprocess.run(command, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            if name == 'train': check_frozen(root, folder)
        result = dict(passed=True, group=group, protocol=protocol,
            best=read(folder / 'train/best_ap70_metrics.json'),
            initial=read(folder / 'train/initial_metrics.json'),
            frozen=read(folder / 'all_frozen_verified.json'), evaluations={})
        checkpoint_sha = sha(folder / 'train/best_ap70.pth')
        for budget in BUDGETS:
            row = evaluation(folder / ('b' + str(budget)))
            assert row['manifest']['checkpoint_sha256'] == checkpoint_sha
            if 'communication_frames' in row:
                assert all(x['bytes'] <= budget for x in row['communication_frames'])
            result['evaluations'][str(budget)] = row
        write_json(folder / 'summary.json', result)
        write_json(status, dict(state='complete', finished_unix=time.time()))
    except Exception:
        write_json(status, dict(state='failed', error=traceback.format_exc(), note='No automatic retry'))
        raise


def launch(root, gpus):
    assert len(gpus) == len(set(gpus)) == 2
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    if shutil.disk_usage(str(root)).free < 12 * 1024**3: raise RuntimeError('Disk reserve too small')
    protocol = read(root / 'protocol.json')
    assert sha(Path(protocol['audit_path'])) == protocol['audit_sha256']
    for path,digest in protocol.get('audits',{}).items(): assert sha(Path(path))==digest
    groups = list(protocol['groups'])
    assert len(groups)==len(gpus)
    admissions = {group: check_gpu(gpu, True, .2) for group, gpu in zip(groups, gpus)}
    setup = dict(gpus=dict(zip(groups, gpus)), admissions=admissions,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
        protocol_sha256=sha(root / 'protocol.json'), started_unix=time.time(), memory_fraction=.2)
    write_json(root / 'launch.json', setup)
    processes = {}
    for group, gpu in zip(groups, gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
            OPENCOOD_CUDA_MEMORY_FRACTION='.2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (root / group / 'pipeline.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_event_round',
                '--root', str(root), '--run-group', group], cwd=str(REPO), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[group] = process.pid
    write_json(root / 'processes.json', processes)
    print(json.dumps(dict(processes=processes, **setup)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--audit-root')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prepare-receivers', action='store_true')
    parser.add_argument('--run-group', choices=list(GROUPS)+list(RECEIVER_GROUPS))
    parser.add_argument('--gpus', type=int, nargs=2, default=[2, 3])
    args = parser.parse_args(); root = Path(args.root)
    if args.prepare_receivers: prepare_receivers(root, Path(args.audit_root))
    elif args.prepare: prepare(root, Path(args.audit_root))
    elif args.run_group: run_group(root, args.run_group)
    else: launch(root, args.gpus)

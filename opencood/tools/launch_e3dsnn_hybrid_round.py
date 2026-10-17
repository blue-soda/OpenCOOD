"""Four-way transplant: full zero-shot gate, then matched ten-epoch adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from opencood.tools.train_e3dsnn_single import write_json


COMBINATIONS = ('ss', 'sa', 'as', 'aa')
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def free_gpu(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits']).decode()
    used, free, util = map(int, row.strip().split(','))
    if used > 1000 or free < 12000 or util > 10:
        raise RuntimeError('GPU {} is not free: {}'.format(gpu, row))


def make_jobs(root, combination, stage):
    root = Path(root)
    config_dir = REPO / 'opencood/hypes_yaml/dair-v2x/snn'
    config = config_dir / ('e3dsnn_hybrid_' + combination + '_181.yaml')
    name = 'hybrid_' + combination + '_' + stage
    base = [sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single', '--workers', '2', '--seed', '20260929']
    checkpoint = root / (combination + '_initial.pth') if stage == 'zero' else root / name / 'best_ap70.pth'
    jobs = []
    if stage == 'adapt10':
        jobs.append(dict(name=name + '_train', timeout_seconds=21600, command=base + ['--mode', 'train',
            '--config', str(config), '--checkpoint', str(root / (combination + '_initial.pth')),
            '--output', str(root / name), '--epochs', '10', '--selection-iou', '0.7']))
    for mode in (('on',) if stage == 'zero' else ('on', 'off')):
        cfg = config if mode == 'on' else config_dir / ('e3dsnn_hybrid_' + combination + '_ego_181.yaml')
        folder = root / (name + '_' + mode)
        command = base + ['--mode', 'eval', '--config', str(cfg), '--checkpoint', str(checkpoint),
                         '--output', str(folder), '--eval-3d', '--save-predictions']
        if mode == 'on': command += ['--packet-roundtrip', '--fusion-diagnostics']
        jobs.append(dict(name=name + '_' + mode, timeout_seconds=3600, command=command))
        jobs.append(dict(name=name + '_' + mode + '_replay', timeout_seconds=3600,
            command=['env', 'CUDA_VISIBLE_DEVICES=', sys.executable, '-u', '-m',
                     'opencood.tools.replay_e3dsnn_predictions', '--evaluation-dir', str(folder)]))
    if stage == 'adapt10':
        jobs.append(dict(name=name + '_diagnostic', timeout_seconds=3600,
            command=['env', 'CUDA_VISIBLE_DEVICES=', sys.executable, '-u', '-m', 'opencood.tools.diagnose_e3dsnn_ap70',
                '--ego-dir', str(root / (name + '_off')), '--fusion-dir', str(root / (name + '_on')),
                '--output', str(root / (name + '_diagnostic.json'))]))
    return name, jobs


def evaluation(folder):
    e, r, index = [read(folder / p) for p in ('evaluation.json', 'offline_replay.json', 'predictions/index.json')]
    assert r['passed'] and r['exact_match'] and r['all_file_hashes_verified']
    assert e['ap'] == r['ap'] and e['ap_3d'] == r['ap_3d']
    assert e['samples'] == len(index['frames']) == 1738 and e['full_split'] and e['gt_boxes'] == 28622
    assert e['predictions']['index_sha256'] == sha(folder / 'predictions/index.json')
    digest = hashlib.sha256()
    for row in index['frames']:
        path = folder / 'predictions' / row['file']
        assert sha(path) == row['sha256']
        with np.load(str(path)) as frame:
            gt = frame['gt_boxes']
            digest.update(row['frame_id'].encode()); digest.update(str(gt.shape).encode()); digest.update(gt.tobytes())
    assert digest.hexdigest() == '54be972f1d9a41bcbf41b8dc234f35db81a7c43ef5ca6958768a7bda27f980ee'
    manifests = list(folder.glob('manifest_*.json')); assert len(manifests) == 1
    result = dict(evaluation=e, replay=r, manifest=read(manifests[0]), ordered_gt_sha256=digest.hexdigest())
    if 'feature_packets' in e:
        rows = [dict(frame_id=f['frame_id'], bytes=f['communication']['total_bytes']) for f in index['frames']]
        assert sum(x['bytes'] for x in rows) == e['feature_packets']['total_bytes']
        result['communication_frames'] = rows
    return result


def summarize(root, source, stage):
    audit = read(root / 'audit.json'); result = dict(stage=stage, combinations={})
    for combo in COMBINATIONS:
        name = 'hybrid_' + combo + '_' + stage
        assert read(root / (name + '_pipeline.json'))['state'] == 'complete'
        on = evaluation(root / (name + '_on'))
        entry = dict(on=on)
        if stage == 'zero':
            assert on['manifest']['checkpoint_sha256'] == audit['combinations'][combo]['checkpoint_sha256']
            if combo in ('ss', 'aa'):
                donor = 'count4' if combo == 'ss' else 'relu'
                old = read(source / (donor + '_aux1_joint30_on') / 'evaluation.json')
                differences = {k: on['evaluation']['ap'][k] - old['ap'][k] for k in old['ap']}
                assert max(abs(v) for v in differences.values()) <= .0002, 'Native AP regression exceeds 0.02pp'
                entry['native_ap_differences'] = differences
        else:
            folder = root / name
            for filename in ('frozen_encoder.json', 'fixed_bn.json'):
                check = read(folder / filename)
                assert check['passed'] and check['verified_through_epoch'] == 10
                assert check['initial_sha256'] == check['current_sha256']
                if filename == 'frozen_encoder.json':
                    assert check['initial_sha256'] == audit['combinations'][combo]['encoder_sha256']
                entry[filename] = check
            assert read(folder / 'status.json')['state'] == 'complete'
            entry['best_metrics'] = read(folder / 'best_ap70_metrics.json')
            entry['checkpoint_sha256'] = sha(folder / 'best_ap70.pth')
            assert entry['checkpoint_sha256'] == on['manifest']['checkpoint_sha256']
            entry['off'] = evaluation(root / (name + '_off'))
            assert entry['off']['manifest']['checkpoint_sha256'] == entry['checkpoint_sha256']
            diagnostic = read(root / (name + '_diagnostic.json')); assert diagnostic['passed']
            entry['gt_transitions70'] = diagnostic['gt_transitions70']
            zero = read(root / 'zero_summary.json')['combinations'][combo]['on']
            entry['packet_bytes_equal_to_zero'] = on['communication_frames'] == zero['communication_frames']
        result['combinations'][combo] = entry
    result['paired_sender_bytes_equal'] = {enc: result['combinations'][enc + 's']['on']['communication_frames'] ==
        result['combinations'][enc + 'a']['on']['communication_frames'] for enc in ('s', 'a')}
    # Byte equality is measured, not assumed: independent sparse CUDA forwards may differ at rounding boundaries.
    result['passed'] = True
    with (root / (stage + '_summary.json')).open('x') as stream: json.dump(result, stream, indent=2)
    return result


def coordinate(root, source):
    setup = read(root / 'launch.json'); status = root / 'round_status.json'
    try:
        for stage in ('zero', 'adapt10'):
            if stage == 'adapt10': assert read(root / 'zero_summary.json')['passed']
            processes = []
            # Check all assigned GPUs before dispatching any new stage.
            for gpu in setup['gpus']: free_gpu(gpu)
            for combo, gpu in zip(COMBINATIONS, setup['gpus']):
                name, jobs = make_jobs(root, combo, stage)
                plan_path = root / (name + '_plan.json')
                with plan_path.open('x') as stream: json.dump(dict(name=name, jobs=jobs, gpu=gpu), stream, indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO), OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
                with (root / (name + '_supervisor.log')).open('xb') as log:
                    process = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan_path)],
                        cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((name, process))
            write_json(status, dict(state='running', stage=stage, supervisors={n: p.pid for n, p in processes}))
            while any(p.poll() is None for _, p in processes):
                for name, p in processes:
                    if p.poll() not in (None, 0): raise RuntimeError('Pipeline failed: ' + name)
                time.sleep(15)
            if any(p.returncode for _, p in processes): raise RuntimeError('A stage failed')
            summarize(root, source, stage)
        write_json(status, dict(state='complete', stages=['zero', 'adapt10']))
    except Exception:
        write_json(status, dict(state='failed', error=traceback.format_exc(),
            note='Other already-launched pipelines are not killed; inspect their status before any recovery.'))
        raise


def launch(root, source, gpus):
    root, source = Path(root).resolve(), Path(source).resolve()
    if len(gpus) != 4 or len(set(gpus)) != 4: raise ValueError('Four distinct GPUs required')
    if (root / 'launch.json').exists(): raise FileExistsError('Already launched; no automatic retry')
    audit = read(root / 'audit.json'); assert audit['passed']
    for combo in COMBINATIONS:
        assert sha(root / (combo + '_initial.pth')) == audit['combinations'][combo]['checkpoint_sha256']
        assert sha(REPO / 'opencood/hypes_yaml/dair-v2x/snn' / ('e3dsnn_hybrid_' + combo + '_181.yaml')) == audit['combinations'][combo]['config_sha256']
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3: raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus: free_gpu(gpu)
    setup = dict(gpus=gpus, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
                 audit_sha256=sha(root / 'audit.json'), started_unix=time.time(), source=str(source))
    with (root / 'launch.json').open('x') as stream: json.dump(setup, stream, indent=2)
    with (root / 'coordinator.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_hybrid_round',
            '--root', str(root), '--source', str(source), '--coordinate'], cwd=str(REPO),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(coordinator_pid=p.pid, **setup)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--gpus', nargs=4, type=int, default=[1, 2, 3, 6])
    parser.add_argument('--coordinate', action='store_true')
    args = parser.parse_args()
    if args.coordinate: coordinate(Path(args.root), Path(args.source))
    else: launch(args.root, args.source, args.gpus)

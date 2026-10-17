"""Gate receiver localization on completed four-way hybrids, then equal budgets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from opencood.tools.launch_e3dsnn_hybrid_round import (
    REPO, read, sha, free_gpu, evaluation, make_jobs as hybrid_jobs)
from opencood.tools.train_e3dsnn_single import write_json


COMBINATIONS = ('ss', 'as', 'sa', 'aa')  # fusion, BEV; encoder is always S


def factorial_contrasts(ap70):
    """Percentage-point differences, not statistical significance or causality."""
    ss, as_, sa, aa = [ap70[k] * 100 for k in ('ss', 'as', 'sa', 'aa')]
    return dict(ann_fusion_with_snn_bev=as_ - ss, ann_fusion_with_ann_bev=aa - sa,
                ann_bev_with_snn_fusion=sa - ss, ann_bev_with_ann_fusion=aa - as_,
                interaction=aa - as_ - sa + ss)


def source_gate(source):
    source = Path(source)
    assert read(source / 'round_status.json')['state'] == 'complete'
    result = read(source / 'adapt10_summary.json')
    assert result['passed']
    for combo in ('ss', 'sa', 'as', 'aa'):
        assert read(source / ('hybrid_' + combo + '_adapt10_pipeline.json'))['state'] == 'complete'
    for combo in ('ss', 'sa'):
        entry = result['combinations'][combo]
        assert sha(source / ('hybrid_' + combo + '_adapt10') / 'best_ap70.pth') == entry['checkpoint_sha256']
    s = result['combinations']['ss']['frozen_encoder.json']
    a = result['combinations']['sa']['frozen_encoder.json']
    assert s['passed'] and a['passed'] and s['current_sha256'] == a['current_sha256']
    return result


def make_jobs(root, combo, stage):
    if combo not in COMBINATIONS or stage not in ('zero', 'adapt10'):
        raise ValueError('Unknown receiver experiment')
    name, jobs = hybrid_jobs(root, combo, stage)
    # Preserve the already-audited train/eval/replay/diagnostic protocol.
    for job in jobs:
        job['name'] = job['name'].replace('hybrid_', 'receiver_')
        job['command'] = [x.replace('e3dsnn_hybrid_' + combo, 'e3dsnn_receiver_' + combo)
                         .replace('hybrid_' + combo + '_' + stage, 'receiver_' + combo + '_' + stage)
                         for x in job['command']]
    return name.replace('hybrid_', 'receiver_'), jobs


def summarize(root, source, stage):
    previous = source_gate(source)
    audit = read(root / 'audit.json')
    result = dict(stage=stage, factor_order=['fusion', 'bev_and_heads'], encoder='fixed_count4',
                  source_summary_sha256=sha(source / 'adapt10_summary.json'), combinations={})
    for combo in COMBINATIONS:
        name = 'receiver_' + combo + '_' + stage
        assert read(root / (name + '_pipeline.json'))['state'] == 'complete'
        on = evaluation(root / (name + '_on'))
        entry = dict(on=on)
        assert on['manifest']['encoder_activation'] == 'count4'
        assert on['manifest']['fusion_activation_kind'] == ('count4' if combo[0] == 's' else 'relu')
        assert on['manifest']['bev_activation'] == ('count4' if combo[1] == 's' else 'relu')
        if stage == 'zero':
            assert on['manifest']['checkpoint_sha256'] == audit['combinations'][combo]['checkpoint_sha256']
            if combo in ('ss', 'aa'):
                donor = 'ss' if combo == 'ss' else 'sa'
                old = previous['combinations'][donor]['on']['evaluation']
                differences = {k: on['evaluation']['ap'][k] - old['ap'][k] for k in old['ap']}
                assert max(abs(v) for v in differences.values()) <= .0002, 'Native AP regression exceeds 0.02pp'
                entry['native_ap_differences'] = differences
        else:
            folder = root / name
            assert read(folder / 'status.json')['state'] == 'complete'
            for filename in ('frozen_encoder.json', 'fixed_bn.json'):
                check = read(folder / filename)
                assert check['passed'] and check['verified_through_epoch'] == 10
                assert check['initial_sha256'] == check['current_sha256']
                if filename == 'frozen_encoder.json':
                    assert check['current_sha256'] == audit['combinations'][combo]['encoder_sha256']
                entry[filename] = check
            entry['best_metrics'] = read(folder / 'best_ap70_metrics.json')
            entry['checkpoint_sha256'] = sha(folder / 'best_ap70.pth')
            assert on['manifest']['checkpoint_sha256'] == entry['checkpoint_sha256']
            entry['off'] = evaluation(root / (name + '_off'))
            assert entry['off']['manifest']['checkpoint_sha256'] == entry['checkpoint_sha256']
            diag = read(root / (name + '_diagnostic.json')); assert diag['passed']
            entry['gt_transitions70'] = diag['gt_transitions70']
            zero = read(root / 'zero_summary.json')['combinations'][combo]['on']
            entry['packet_bytes_equal_to_zero'] = on['communication_frames'] == zero['communication_frames']
        result['combinations'][combo] = entry
    packets = result['combinations']['ss']['on']['communication_frames']
    result['same_sender_packet_bytes_equal'] = {
        c: result['combinations'][c]['on']['communication_frames'] == packets for c in COMBINATIONS}
    result['ap70_contrasts_percentage_points'] = factorial_contrasts({
        c: result['combinations'][c]['on']['evaluation']['ap']['0.7'] for c in COMBINATIONS})
    result['interpretation_limit'] = ('Single-seed validation-set module/weight transplants with joint receiver adaptation; '
                                       'BEV factor includes all heads. No isolated activation causality or significance claim.')
    result['passed'] = True
    with (root / (stage + '_summary.json')).open('x') as f: json.dump(result, f, indent=2)
    return result


def coordinate(root, source):
    setup = read(root / 'launch.json')
    try:
        source_gate(source)
        for stage in ('zero', 'adapt10'):
            if stage == 'adapt10': assert read(root / 'zero_summary.json')['passed']
            for gpu in setup['gpus']: free_gpu(gpu)
            processes = []
            for combo, gpu in zip(COMBINATIONS, setup['gpus']):
                name, jobs = make_jobs(root, combo, stage)
                plan = root / (name + '_plan.json')
                with plan.open('x') as f: json.dump(dict(name=name, jobs=jobs, gpu=gpu), f, indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           NUMPY_MADVISE_HUGEPAGE='0')
                with (root / (name + '_supervisor.log')).open('xb') as log:
                    p = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan)],
                        cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((name, p))
            write_json(root / 'round_status.json', dict(state='running', stage=stage,
                                                       supervisors={n: p.pid for n, p in processes}))
            while any(p.poll() is None for _, p in processes):
                for name, p in processes:
                    if p.poll() not in (None, 0): raise RuntimeError('Pipeline failed: ' + name)
                time.sleep(15)
            if any(p.returncode for _, p in processes): raise RuntimeError('A pipeline failed')
            summarize(root, source, stage)
        write_json(root / 'round_status.json', dict(state='complete', stages=['zero', 'adapt10']))
    except Exception:
        write_json(root / 'round_status.json', dict(state='failed', error=traceback.format_exc(),
                   note='Other launched pipelines are not killed. No automatic retry.'))
        raise


def launch(root, source, gpus):
    root, source = Path(root).resolve(), Path(source).resolve()
    if len(gpus) != 4 or len(set(gpus)) != 4: raise ValueError('Four distinct GPUs required')
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    source_gate(source)
    audit = read(root / 'audit.json'); assert audit['passed']
    assert audit['source_summary_sha256'] == sha(source / 'adapt10_summary.json')
    for c in COMBINATIONS:
        assert sha(root / (c + '_initial.pth')) == audit['combinations'][c]['checkpoint_sha256']
        for suffix in ('_181.yaml', '_ego_181.yaml'):
            path = REPO / 'opencood/hypes_yaml/dair-v2x/snn' / ('e3dsnn_receiver_' + c + suffix)
            assert sha(path) == audit['combinations'][c]['config_sha256'][suffix]
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3: raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus: free_gpu(gpu)
    shutil.copyfile(str(source / 'supervise.py'), str(root / 'supervise.py'))
    setup = dict(gpus=gpus, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
                 audit_sha256=sha(root / 'audit.json'), source=str(source), started_unix=time.time())
    with (root / 'launch.json').open('x') as f: json.dump(setup, f, indent=2)
    with (root / 'coordinator.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_receiver_round',
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

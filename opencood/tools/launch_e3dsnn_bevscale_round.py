"""Fixed/learnable BEV step x high/low LR, from one all-SNN checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu, evaluation, make_jobs as hybrid_jobs
from opencood.tools.train_e3dsnn_single import write_json


GROUPS = ('fixed_high', 'learned_high', 'fixed_low', 'learned_low')


def source_gate(source):
    source = Path(source)
    assert read(source / 'round_status.json')['state'] == 'complete'
    completed = read(source / 'adapt10_summary.json'); assert completed['passed']
    for combo in ('ss', 'as', 'sa', 'aa'):
        assert read(source / ('receiver_' + combo + '_adapt10_pipeline.json'))['state'] == 'complete'
    # Retain the established best all-SNN model instead of the worse extra-LR-reset control.
    prior = Path(read(source / 'launch.json')['source'])
    assert read(prior / 'round_status.json')['state'] == 'complete'
    parent = read(prior / 'adapt10_summary.json'); assert parent['passed']
    initializer = prior / 'hybrid_ss_adapt10/best_ap70.pth'
    reference = parent['combinations']['ss']
    assert sha(initializer) == reference['checkpoint_sha256']
    for entry in completed['combinations'].values():
        assert entry['frozen_encoder.json']['current_sha256'] == reference['frozen_encoder.json']['current_sha256']
    return initializer, reference


def make_jobs(root, group, stage):
    if group not in GROUPS or stage not in ('zero', 'adapt10'): raise ValueError('Unknown scale experiment')
    name, jobs = hybrid_jobs(root, group, stage)
    for job in jobs:
        job['name'] = job['name'].replace('hybrid_', 'bevscale_')
        job['command'] = [x.replace('e3dsnn_hybrid_' + group, 'e3dsnn_bevscale_' + group)
                         .replace('hybrid_' + group + '_' + stage, 'bevscale_' + group + '_' + stage)
                         for x in job['command']]
        job['command'] = [str(Path(root) / 'initial.pth') if x == str(Path(root) / (group + '_initial.pth')) else x
                          for x in job['command']]
    return name.replace('hybrid_', 'bevscale_'), jobs


def contrasts(values):
    return dict(learned_at_high_lr=(values['learned_high'] - values['fixed_high']) * 100,
                learned_at_low_lr=(values['learned_low'] - values['fixed_low']) * 100,
                low_lr_fixed=(values['fixed_low'] - values['fixed_high']) * 100,
                low_lr_learned=(values['learned_low'] - values['learned_high']) * 100)


def summarize(root, source, stage):
    _, reference = source_gate(source)
    audit = read(root / 'audit.json')
    result = dict(stage=stage, source_summary_sha256=sha(source / 'adapt10_summary.json'),
                  initializer_sha256=audit['initializer_sha256'], groups={})
    for group in GROUPS:
        name = 'bevscale_' + group + '_' + stage
        assert read(root / (name + '_pipeline.json'))['state'] == 'complete'
        on = evaluation(root / (name + '_on')); entry = dict(on=on)
        assert on['manifest']['bev_scale_mode'] == group.split('_')[0]
        assert on['manifest']['encoder_activation'] == on['manifest']['fusion_activation_kind'] == 'count4'
        if stage == 'zero':
            assert on['manifest']['checkpoint_sha256'] == audit['initial_state_sha256']
            differences = {k: on['evaluation']['ap'][k] - reference['on']['evaluation']['ap'][k]
                           for k in on['evaluation']['ap']}
            assert max(abs(v) for v in differences.values()) <= .0002, 'Unit-scale native AP mismatch'
            entry['native_ap_differences'] = differences
        else:
            folder = root / name
            assert read(folder / 'status.json')['state'] == 'complete'
            for filename in ('frozen_encoder.json', 'fixed_bn.json'):
                check = read(folder / filename)
                assert check['passed'] and check['verified_through_epoch'] == 10
                assert check['initial_sha256'] == check['current_sha256']
                if filename == 'frozen_encoder.json':
                    assert check['current_sha256'] == audit['encoder_sha256']
                entry[filename] = check
            entry['best_metrics'] = read(folder / 'best_ap70_metrics.json')
            entry['last_scale_report'] = read(folder / 'bev_scales.json')
            assert entry['last_scale_report']['passed'] and entry['last_scale_report']['verified_through_epoch'] == 10
            if group.startswith('fixed'): assert entry['last_scale_report']['changed_from_one'] == 0
            entry['checkpoint_sha256'] = sha(folder / 'best_ap70.pth')
            assert on['manifest']['checkpoint_sha256'] == entry['checkpoint_sha256']
            entry['off'] = evaluation(root / (name + '_off'))
            assert entry['off']['manifest']['checkpoint_sha256'] == entry['checkpoint_sha256']
            diag = read(root / (name + '_diagnostic.json')); assert diag['passed']
            entry['gt_transitions70'] = diag['gt_transitions70']
            entry['best_scale_report'] = entry['best_metrics']['bev_scale_report']
            entry['packet_bytes_equal_to_zero'] = on['communication_frames'] == read(root / 'zero_summary.json')['groups'][group]['on']['communication_frames']
        result['groups'][group] = entry
    result['ap70_contrasts_percentage_points'] = contrasts({g: result['groups'][g]['on']['evaluation']['ap']['0.7'] for g in GROUPS})
    result['same_sender_packet_bytes_equal'] = {g: result['groups'][g]['on']['communication_frames'] ==
        result['groups']['fixed_high']['on']['communication_frames'] for g in GROUPS}
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
            for group, gpu in zip(GROUPS, setup['gpus']):
                name, jobs = make_jobs(root, group, stage)
                plan = root / (name + '_plan.json')
                with plan.open('x') as f: json.dump(dict(name=name, jobs=jobs, gpu=gpu), f, indent=2)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
                with (root / (name + '_supervisor.log')).open('xb') as log:
                    p = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan)], cwd=str(REPO),
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((name, p))
            write_json(root / 'round_status.json', dict(state='running', stage=stage, supervisors={n: p.pid for n, p in processes}))
            while any(p.poll() is None for _, p in processes):
                for name, p in processes:
                    if p.poll() not in (None, 0): raise RuntimeError('Pipeline failed: ' + name)
                time.sleep(15)
            if any(p.returncode for _, p in processes): raise RuntimeError('A pipeline failed')
            summarize(root, source, stage)
        write_json(root / 'round_status.json', dict(state='complete', stages=['zero', 'adapt10']))
    except Exception:
        write_json(root / 'round_status.json', dict(state='failed', error=traceback.format_exc(),
                   note='No automatic retry; other already-launched pipelines are not killed.'))
        raise


def launch(root, source, gpus):
    root, source = Path(root).resolve(), Path(source).resolve()
    if len(gpus) != 4 or len(set(gpus)) != 4: raise ValueError('Four distinct GPUs required')
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    source_gate(source)
    audit = read(root / 'audit.json'); assert audit['passed']
    assert audit['source_summary_sha256'] == sha(source / 'adapt10_summary.json')
    assert sha(root / 'initial.pth') == audit['initial_state_sha256']
    for group in GROUPS:
        for suffix in ('_181.yaml', '_ego_181.yaml'):
            path = REPO / 'opencood/hypes_yaml/dair-v2x/snn' / ('e3dsnn_bevscale_' + group + suffix)
            assert sha(path) == audit['groups'][group]['config_sha256'][suffix]
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3: raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus: free_gpu(gpu)
    shutil.copyfile(str(source / 'supervise.py'), str(root / 'supervise.py'))
    setup = dict(gpus=gpus, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
                 audit_sha256=sha(root / 'audit.json'), source=str(source), started_unix=time.time())
    with (root / 'launch.json').open('x') as f: json.dump(setup, f, indent=2)
    with (root / 'coordinator.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_bevscale_round',
            '--root', str(root), '--source', str(source), '--coordinate'], cwd=str(REPO), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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

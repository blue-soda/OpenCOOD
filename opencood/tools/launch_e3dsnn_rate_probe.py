"""Bounded fixed-checkpoint communication probes; no training or AP tuning."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu, evaluation
from opencood.tools.train_e3dsnn_single import write_json
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import quantized_feature_packet as codec


FAMILIES = ('snn_fusion', 'ann_receiver', 'ann_full')


def variants(family):
    if family not in FAMILIES: raise ValueError('Unknown family')
    if family == 'ann_full':
        return [('native', None)] + [('q%d' % b, dict(bits=b, keep_fraction=1.)) for b in (8, 4, 3, 2)]
    return [('native', None)] + [('keep%d' % k, dict(bits=3, keep_fraction=k / 100.)) for k in (100, 50, 25)]


def sources(base):
    base = Path(base)
    definitions = dict(snn_fusion=('receiver', 'sa', 50), ann_receiver=('hybrid', 'sa', 40),
                       ann_full=('hybrid', 'aa', 40))
    result = {}
    for family, (round_name, combo, budget) in definitions.items():
        source = base / ('e3dsnn_%s_round1_20261017' % round_name)
        assert read(source / 'round_status.json')['state'] == 'complete'
        summary = read(source / 'adapt10_summary.json'); assert summary['passed']
        entry = summary['combinations'][combo]
        checkpoint = source / ('%s_%s_adapt10' % (round_name, combo)) / 'best_ap70.pth'
        assert sha(checkpoint) == entry['checkpoint_sha256']
        result[family] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            config=str(REPO / 'opencood/hypes_yaml/dair-v2x/snn' /
                       ('e3dsnn_%s_%s_181.yaml' % (round_name, combo))),
            source_summary_sha256=sha(source / 'adapt10_summary.json'),
            reference_ap=entry['on']['evaluation']['ap'],
            reference_communication=entry['on']['evaluation']['feature_packets'],
            adaptation_epochs=budget)
    return result


def audit(root, base):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root / 'audit.json').exists(): raise FileExistsError('Audit already completed')
    torch.manual_seed(20260929); np.random.seed(20260929)
    source = sources(base)
    report = dict(sources=source, families={}, scope='Two validation frames, cached sender, discarded outputs')
    for family in FAMILIES:
        cfg = load_yaml(source[family]['config'])
        dataset = build_dataset(cfg, train=False)
        data = train_utils.to_device(dataset.collate_batch_test([dataset[0]]), torch.device('cuda'))['ego']
        # Batch-one reader is audited on two independently collated validation frames.
        model = train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(source[family]['checkpoint'], map_location='cpu'), strict=True)
        report['families'][family] = dict(configs={}, frames=[])
        raw_cfg = yaml.safe_load(Path(source[family]['config']).read_text())
        for name, options in variants(family):
            fresh = dict(raw_cfg)
            fresh['model'] = dict(raw_cfg['model'], args=dict(raw_cfg['model']['args']))
            fresh['model']['args']['message_codec'] = options
            path = root / (family + '_' + name + '.yaml')
            with path.open('x') as f: yaml.safe_dump(fresh, f)
            report['families'][family]['configs'][name] = dict(path=str(path), sha256=sha(path), options=options)
        original_encode = model.encode_lidar
        for index in (0, 1):
            data = train_utils.to_device(dataset.collate_batch_test([dataset[index]]), torch.device('cuda'))['ego']
            with torch.no_grad():
                cached = {id(data[k]): original_encode(data[k], 1) for k in ('processed_lidar', 'road_lidar')}
                model.encode_lidar = lambda lidar, n: cached[id(lidar)]
                model.message_codec = None; model.packet_roundtrip = True
                reference = model(data)
                road = cached[id(data['road_lidar'])]
                c = road.indices[:, 1:].cpu().numpy(); f = road.features.detach().cpu().numpy()
                frame = dict(frame_id=data['frame_ids'][0], variants={})
                for name, options in variants(family):
                    model.message_codec = options; model.packet_roundtrip = True
                    output = model(data)
                    if options is None:
                        assert all(torch.equal(reference[k], output[k]) for k in ('psm', 'rm', 'dm'))
                    else:
                        coords, codes, scales = codec.prepare(c, f, model.activation, options)
                        dense = torch.zeros_like(road.dense())
                        xyz = torch.as_tensor(coords.astype('int64'), device='cuda')
                        dense[0, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = torch.as_tensor(
                            (codes.astype(np.float32) * scales).T.copy(), device='cuda')
                        cached[id(data['road_lidar'])] = SimpleNamespace(dense=lambda: dense)
                        model.message_codec = None; model.packet_roundtrip = False
                        oracle = model(data)
                        cached[id(data['road_lidar'])] = road
                        assert all(torch.equal(oracle[k], output[k]) for k in ('psm', 'rm', 'dm'))
                        if model.activation == 'count4' and options['keep_fraction'] == 1.:
                            assert all(torch.equal(reference[k], output[k]) for k in ('psm', 'rm', 'dm'))
                    assert all(torch.isfinite(output[k]).all() for k in ('psm', 'rm', 'dm'))
                    frame['variants'][name] = dict(packet_bytes=output['packet_bytes'][0], oracle_exact=True)
                report['families'][family]['frames'].append(frame)
        model.encode_lidar = original_encode
        del model, original_encode, cached, road, dataset, data, output, reference
        gc.collect(); torch.cuda.empty_cache()
    report['passed'] = True
    with (root / 'audit.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


def run_family(root, family):
    root = Path(root); audit_result = read(root / 'audit.json'); assert audit_result['passed']
    source = audit_result['sources'][family]
    assert sha(source['checkpoint']) == source['checkpoint_sha256']
    summary = dict(family=family, source=source, variants={})
    try:
        for name, options in variants(family):
            item = audit_result['families'][family]['configs'][name]
            assert sha(item['path']) == item['sha256']
            out = root / (family + '_' + name)
            if out.exists(): raise FileExistsError('No automatic overwrite/retry: ' + str(out))
            write_json(root / (family + '_status.json'), dict(state='running', variant=name, updated_unix=time.time()))
            commands = [([sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single',
                '--mode', 'eval', '--config', item['path'], '--checkpoint', source['checkpoint'],
                '--output', str(out), '--workers', '2', '--seed', '20260929', '--eval-3d',
                '--save-predictions', '--packet-roundtrip'], False),
                ([sys.executable, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                  '--evaluation-dir', str(out)], True)]
            for command, cpu in commands:
                env = dict(os.environ)
                if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
                with (root / (family + '_' + name + ('_replay' if cpu else '') + '.log')).open('xb') as log:
                    # subprocess.run enforces a per-job deadline and reaps only its own child.
                    subprocess.run(command, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
            entry = evaluation(out)
            assert entry['manifest']['checkpoint_sha256'] == source['checkpoint_sha256']
            assert entry['manifest']['message_codec'] == options
            if name == 'native' or (family != 'ann_full' and name == 'keep100'):
                differences = {k: entry['evaluation']['ap'][k] - source['reference_ap'][k] for k in source['reference_ap']}
                assert max(abs(v) for v in differences.values()) <= .0002, 'Native/lossless AP gate failed'
                entry['native_ap_differences'] = differences
            summary['variants'][name] = entry
            write_json(root / (family + '_partial.json'), summary)
        summary['passed'] = True
        write_json(root / (family + '_summary.json'), summary)
        write_json(root / (family + '_status.json'), dict(state='complete', updated_unix=time.time()))
    except Exception:
        write_json(root / (family + '_status.json'), dict(state='failed', error=traceback.format_exc(),
            note='No retry; other families and existing training are not terminated.'))
        raise


def launch(root, base, gpus):
    root = Path(root); report = read(root / 'audit.json'); assert report['passed']
    assert sources(base) == report['sources']
    if len(set(gpus)) != 3: raise ValueError('Three distinct GPUs required')
    if (root / 'launch.json').exists(): raise FileExistsError('Already attempted')
    if shutil.disk_usage(str(root)).free < 8 * 1024 ** 3: raise RuntimeError('Insufficient disk reserve')
    for gpu in gpus: free_gpu(gpu)
    setup = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip(),
                 audit_sha256=sha(root / 'audit.json'), gpus=gpus, started_unix=time.time())
    with (root / 'launch.json').open('x') as f: json.dump(setup, f, indent=2)
    processes = {}
    for family, gpu in zip(FAMILIES, gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
        with (root / (family + '_supervisor.log')).open('xb') as log:
            p = subprocess.Popen([sys.executable, '-u', '-m', 'opencood.tools.launch_e3dsnn_rate_probe',
                '--root', str(root), '--family', family], cwd=str(REPO), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[family] = dict(pid=p.pid, gpu=gpu)
    write_json(root / 'processes.json', processes)
    print(json.dumps(processes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--base', default='/data1/tzm/workspace/diagnostics')
    parser.add_argument('--gpus', type=int, nargs=3, default=[0, 4, 5])
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    if args.audit: audit(args.root, args.base)
    elif args.family: run_family(args.root, args.family)
    else: launch(args.root, args.base, args.gpus)

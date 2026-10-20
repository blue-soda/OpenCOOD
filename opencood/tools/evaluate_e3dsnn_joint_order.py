"""Fixed-checkpoint learned versus L1 order; same rows, measured (not equal) bytes.

No training or checkpoint selection. Real batch audit precedes six full-split
wire evaluations and independent CPU replay. Failures are not retried.
"""
import argparse
import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, evaluation
from opencood.tools.train_e3dsnn_single import write_json


def run(root, source):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Fresh diagnostic directory required')
    try:
        write_json(root / 'status.json', dict(state='auditing', started_unix=time.time()))
        prior = read(source / 'adapt10_summary.json')
        assert prior['passed']
        checkpoint = source / 'joint_adapt10/best_ap70.pth'
        digest = sha(checkpoint)
        assert digest == prior['groups']['joint']['checkpoint_sha256']
        raw = yaml.safe_load((source / 'joint/p3.yaml').read_text())
        configs = {}
        for order in ('learned', 'l1'):
            for prefix in (1, 2, 3):
                key = '{}_p{}'.format(order, prefix)
                cfg = copy.deepcopy(raw)
                cfg['model']['args'].update(joint_eval_order=order, readout_prefix=prefix,
                                           content_full_packet=False)
                path = root / (key + '.yaml')
                path.write_text(yaml.safe_dump(cfg))
                configs[key] = sha(path)
        random.seed(20260929)
        np.random.seed(20260929)
        torch.manual_seed(20260929)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg = load_yaml(str(root / 'learned_p3.yaml'))
        dataset = build_dataset(cfg, train=True)
        data = train_utils.to_device(
            dataset.collate_batch_train([dataset[0], dataset[1]]),
            torch.device('cuda'))['ego']
        model = train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
        audit = dict(checkpoint_sha256=digest, config_hashes=configs,
                     frames=data['frame_ids'], train=False, comparisons={},
                     code_commit=subprocess.check_output(
                         ['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip())
        with torch.no_grad():
            ego, membrane = model.encode_membrane(data['processed_lidar'], 2)
            road = model.encode_lidar(data['road_lidar'], 2)
            model.encode_membrane = lambda lidar, n: (ego, membrane)
            model.encode_lidar = lambda lidar, n: road
            from opencood.models.fuse_modules.e3dsnn_progressive_sender import l1_partition
            counts = model.content.encode(model.select_road(road, 2)[0:1]).counts
            learned_parts = model.progressive_sender(counts, 3)
            l1_parts = l1_partition(counts, 3)
            assert torch.equal(sum(learned_parts), counts)
            assert torch.equal(sum(l1_parts), counts)
            assert [int((x != 0).any(1).sum()) for x in learned_parts] == [
                int((x != 0).any(1).sum()) for x in l1_parts]
            audit['same_complete_content_and_rows_per_batch'] = True
            final = {}
            for order in ('learned', 'l1'):
                model.joint_eval_order = order
                for prefix in (1, 2, 3):
                    model.readout_prefix = prefix
                    model.packet_roundtrip = False
                    direct = model(data)
                    model.packet_roundtrip = True
                    wire = model(data)
                    assert all(torch.equal(direct[k], wire[k]) for k in ('psm', 'rm', 'dm'))
                    audit['comparisons']['{}_p{}'.format(order, prefix)] = dict(
                        direct_wire_exact=True, bytes=wire['packet_bytes'])
                final[order] = wire
            # Different accumulation orders can cross a quantizer boundary through
            # roundoff. Record prediction differences, do not assert bit identity.
            audit['p3_order_prediction_max_difference'] = {
                k: float((final['learned'][k] - final['l1'][k]).abs().max())
                for k in ('psm', 'rm', 'dm')}
        audit['passed'] = True
        write_json(root / 'audit.json', audit)
        del model, dataset, data, ego, membrane, road, counts, learned_parts, l1_parts, direct, wire, final
        torch.cuda.empty_cache()
        report = dict(checkpoint_sha256=digest, audit_sha256=sha(root / 'audit.json'),
                      results={}, passed=False)
        for key in configs:
            output = root / key
            write_json(root / 'status.json', dict(state='evaluating', current=key))
            command = [sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single',
                       '--workers', '2', '--seed', '20260929', '--mode', 'eval',
                       '--config', str(root / (key + '.yaml')), '--checkpoint', str(checkpoint),
                       '--output', str(output), '--eval-3d', '--save-predictions',
                       '--profile', '--packet-roundtrip']
            with (root / (key + '.log')).open('xb') as log:
                subprocess.run(command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=3600)
            with (root / (key + '_replay.log')).open('xb') as log:
                subprocess.run([sys.executable, '-u', '-m', 'opencood.tools.replay_e3dsnn_predictions',
                                '--evaluation-dir', str(output)], cwd=str(REPO),
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=3600)
            entry = evaluation(output)
            assert entry['manifest']['checkpoint_sha256'] == digest
            assert entry['manifest']['config_sha256'] == configs[key]
            reference = prior['groups']['joint']['prefixes'][key.split('_')[1]]
            assert entry['ordered_gt_sha256'] == reference['ordered_gt_sha256']
            if key.startswith('learned'):
                assert max(abs(entry['evaluation']['ap'][k] - reference['evaluation']['ap'][k])
                           for k in ('0.3', '0.5', '0.7')) <= .0002
                assert entry['communication_frames'] == reference['communication_frames']
            report['results'][key] = entry
            write_json(root / 'summary.json', report)
        assert sha(checkpoint) == digest
        report['passed'] = True
        write_json(root / 'summary.json', report)
        write_json(root / 'status.json', dict(state='complete', finished_unix=time.time()))
    except Exception:
        write_json(root / 'status.json', dict(state='failed', error=traceback.format_exc(),
                                             note='No automatic retry'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', default='/data1/tzm/workspace/diagnostics/e3dsnn_joint_round1_20261019')
    args = parser.parse_args()
    run(Path(args.root), Path(args.source))

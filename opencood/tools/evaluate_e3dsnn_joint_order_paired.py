"""Paired ordering evaluation: one encoder/message realization per frame.

Supersedes independent CUDA passes for causal order comparison; never overwrites
their failed equality check. All six treatments reuse the same encoded tensors.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import write_json, seed_worker
from opencood.tools.launch_e3dsnn_hybrid_round import evaluation, sha, read
from opencood.utils import eval_utils, eval_3d_utils


def run(root, source, previous, audit_only=False):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Fresh paired evaluation directory required')
    try:
        prior = read(source / 'adapt10_summary.json')
        assert prior['passed'] and read(previous / 'audit.json')['passed']
        checkpoint = source / 'joint_adapt10/best_ap70.pth'
        digest = sha(checkpoint)
        assert digest == prior['groups']['joint']['checkpoint_sha256']
        random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
        torch.cuda.set_per_process_memory_fraction(.2)
        cfg = load_yaml(str(previous / 'learned_p3.yaml'))
        dataset = build_dataset(cfg, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=dataset.collate_batch_test, worker_init_fn=seed_worker,
            multiprocessing_context='spawn')
        model = train_utils.create_model(cfg).cuda().eval()
        model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
        model.packet_roundtrip = True
        keys = ['{}_p{}'.format(o, p) for o in ('learned', 'l1') for p in (1, 2, 3)]
        stats = {k: {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)} for k in keys}
        stats3d = copy.deepcopy(stats)
        rows = {k: [] for k in keys}
        source_hashes = []
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        for key in keys:
            folder = root / key
            (folder / 'predictions').mkdir(parents=True)
            config = previous / (key + '.yaml')
            (folder / 'config.yaml').write_bytes(config.read_bytes())
            write_json(folder / 'manifest_paired.json', dict(
                git_commit=commit, checkpoint_sha256=digest, config_sha256=sha(config),
                sender_ranking=key.split('_')[0], paired_shared_encoder_and_content=True,
                paired_scope='Identical cached sparse ego, continuous membrane, road and Count3 content per frame',
                timing_scope='No standalone latency claim from paired cached evaluation'))
        encode_ego, encode_road = model.encode_membrane, model.encode_lidar
        content_encode = model.content.encode
        sender = model.progressive_sender.forward
        started = time.time()
        write_json(root / 'status.json', dict(state='paired_inference', frames=0,
                                             audit_only=audit_only))
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if time.time() - started > 7200:
                    raise TimeoutError('Paired inference exceeded two hours')
                batch = train_utils.to_device(batch, torch.device('cuda'))
                data = batch['ego']
                ego, membrane = encode_ego(data['processed_lidar'], 1)
                road = encode_road(data['road_lidar'], 1)
                message = content_encode(model.select_road(road, 1))
                learned_chunks = sender(message.counts, 3)
                from opencood.models.fuse_modules.e3dsnn_progressive_sender import l1_partition
                l1_chunks = l1_partition(message.counts, 3)
                assert torch.equal(sum(learned_chunks), message.counts)
                assert torch.equal(sum(l1_chunks), message.counts)
                assert [int((x != 0).any(1).sum()) for x in learned_chunks] == [
                    int((x != 0).any(1).sum()) for x in l1_chunks]
                model.encode_membrane = lambda lidar, n: (ego, membrane)
                model.encode_lidar = lambda lidar, n: road
                model.content.encode = lambda features: message
                model.progressive_sender.forward = lambda counts, packets: learned_chunks
                fid = str(dataset.data[i])
                source_hashes.append(dict(frame_id=fid, count3_sha256=hashlib.sha256(
                    message.counts.cpu().numpy().tobytes()).hexdigest()))
                for key in keys:
                    order, prefix = key.split('_')
                    model.joint_eval_order = order
                    model.readout_prefix = int(prefix[1:])
                    output = model(data)
                    if audit_only:
                        model.packet_roundtrip = False
                        direct = model(data)
                        model.packet_roundtrip = True
                        assert all(torch.equal(direct[k], output[k]) for k in ('psm','rm','dm'))
                    boxes, scores, gt = dataset.post_process_no_fusion(batch, {'ego': output})
                    for threshold in stats[key]:
                        eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[key], threshold)
                    eval_3d_utils.calculate_tp_fp_3d(boxes, scores, gt, stats3d[key])
                    pred = boxes.cpu().numpy() if boxes is not None else np.empty((0,8,3), dtype=np.float32)
                    score = scores.cpu().numpy() if scores is not None else np.empty((0,), dtype=np.float32)
                    if not np.isfinite(pred).all() or not np.isfinite(score).all():
                        raise ValueError('Nonfinite predictions')
                    path = root / key / 'predictions' / ('{:06d}.npz'.format(i))
                    np.savez_compressed(str(path), frame_id=np.asarray(fid), pred_boxes=pred,
                                        scores=score, gt_boxes=gt.cpu().numpy())
                    lengths = output['packet_bytes']
                    rows[key].append(dict(index=i, frame_id=fid, file=path.name, sha256=sha(path),
                        predicted_boxes=len(pred), gt_boxes=len(gt),
                        communication=dict(road_to_vehicle_packet_bytes=lengths, total_bytes=sum(lengths))))
                model.encode_membrane, model.encode_lidar = encode_ego, encode_road
                model.content.encode = content_encode
                model.progressive_sender.forward = sender
                for order in ('learned', 'l1'):
                    sizes = [rows[order+'_p'+str(p)][-1]['communication']['total_bytes']
                             for p in (1,2,3)]
                    assert sizes == sorted(sizes)
                if audit_only and i == 1:
                    break
                if (i+1) % 100 == 0:
                    write_json(root / 'status.json', dict(state='paired_inference', frames=i+1, total=len(dataset)))
                    print(json.dumps(dict(frames=i+1)), flush=True)
        if audit_only:
            assert len(source_hashes) == 2
            assert sha(checkpoint) == digest
            write_json(root / 'audit.json', dict(passed=True, frames=source_hashes,
                checkpoint_sha256=digest, code_commit=commit,
                direct_wire_exact_all_six=True, shared_content_and_balanced_rows_both_frames=True,
                bytes={k:[x['communication']['total_bytes'] for x in v] for k,v in rows.items()}))
            write_json(root / 'status.json', dict(state='audit_complete'))
            return
        assert len(source_hashes) == 1738
        write_json(root / 'shared_content_hashes.json', source_hashes)
        report = dict(passed=False, checkpoint_sha256=digest, results={},
                      shared_content_sha256=sha(root / 'shared_content_hashes.json'),
                      historical_byte_differences={})
        for key in keys:
            folder = root / key
            frame_rows = rows[key]
            write_json(folder / 'predictions/index.json', dict(schema=1, frames=frame_rows,
                       order='validation loader order', metric='BEV and supplemental 3D'))
            lengths = [x['communication']['total_bytes'] for x in frame_rows]
            metrics = dict(samples=len(frame_rows), indexed_samples=len(dataset), full_split=True,
                predicted_boxes=sum(x['predicted_boxes'] for x in frame_rows),
                gt_boxes=sum(x['gt_boxes'] for x in frame_rows),
                ap={str(t): eval_utils.calculate_ap(stats[key], t)[0] for t in stats[key]},
                ap_3d={str(t): eval_utils.calculate_ap(stats3d[key], t)[0] for t in stats3d[key]},
                ap_convention='VOC2010 precision-envelope integral; not KITTI R11/R40',
                metric='BEV and upright 3D AP; identical cached encoder/content for ordering comparison',
                predictions=dict(directory='predictions', frames=len(frame_rows),
                    index_sha256=sha(folder / 'predictions/index.json')),
                feature_packets=dict(count=len(lengths), mean_bytes=float(np.mean(lengths)),
                    median_bytes=float(np.median(lengths)), min_bytes=min(lengths), max_bytes=max(lengths),
                    p95_bytes=float(np.percentile(lengths,95)), total_bytes=sum(lengths),
                    scope='serialized road payload and metadata; excludes network framing/retransmission'))
            write_json(folder / 'evaluation.json', metrics)
            write_json(root / 'status.json', dict(state='cpu_replay', current=key))
            with (root / (key + '_replay.log')).open('xb') as log:
                subprocess.run([sys.executable, '-m', 'opencood.tools.replay_e3dsnn_predictions',
                    '--evaluation-dir', str(folder)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                    stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
            entry = evaluation(folder)
            assert entry['manifest']['checkpoint_sha256'] == digest
            reference = prior['groups']['joint']['prefixes'][key.split('_')[1]]
            assert entry['ordered_gt_sha256'] == reference['ordered_gt_sha256']
            if key.startswith('learned'):
                assert max(abs(metrics['ap'][t] - reference['evaluation']['ap'][t])
                           for t in ('0.3','0.5','0.7')) <= .0002
                report['historical_byte_differences'][key] = [
                    dict(frame_id=a['frame_id'], current=a['bytes'], historical=b['bytes'])
                    for a,b in zip(entry['communication_frames'],reference['communication_frames'])
                    if a != b]
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', default='/data1/tzm/workspace/diagnostics/e3dsnn_joint_round1_20261019')
    parser.add_argument('--previous', default='/data1/tzm/workspace/diagnostics/e3dsnn_joint_order_20261020')
    parser.add_argument('--audit-only', action='store_true')
    args=parser.parse_args()
    run(Path(args.root), Path(args.source), Path(args.previous), args.audit_only)

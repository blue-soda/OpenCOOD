"""Join verified SNN/ANN frame packet sizes into a reproducible JSON report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_run(folder, expected_frames):
    folder = Path(folder)
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    evaluation, replay = read('evaluation.json'), read('offline_replay.json')
    if read('status.json')['state'] != 'complete' or not replay.get('passed'):
        raise ValueError('Evaluation and offline replay must complete first')
    if not replay.get('all_file_hashes_verified') or not replay.get('exact_match'):
        raise ValueError('Offline replay must verify hashes and exact AP')
    for key in ('ap', 'ap_3d', 'samples', 'predicted_boxes', 'gt_boxes'):
        if evaluation.get(key) != replay.get(key):
            raise ValueError('Replay differs from evaluation: ' + key)
    index = folder / 'predictions/index.json'
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    if index_hash != evaluation['predictions']['index_sha256']:
        raise ValueError('Prediction index hash mismatch')
    rows = json.loads(index.read_text(encoding='utf-8'))['frames']
    if len(rows) != expected_frames or evaluation['samples'] != expected_frames:
        raise ValueError('Incomplete validation split')
    ids, sizes = [], []
    for i, row in enumerate(rows):
        if row['index'] != i:
            raise ValueError('Noncontiguous frame indices')
        lengths = row['communication']['road_to_vehicle_packet_bytes']
        if len(lengths) != 1 or type(lengths[0]) is not int or lengths[0] <= 0:
            raise ValueError('Expected exactly one positive integer packet size per frame')
        if row['communication']['total_bytes'] != lengths[0]:
            raise ValueError('Incorrect frame byte total')
        ids.append(row['frame_id'])
        sizes.append(lengths[0])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate frame IDs')
    stats = dict(count=len(sizes), total_bytes=sum(sizes), min_bytes=min(sizes),
                 max_bytes=max(sizes), mean_bytes=float(np.mean(sizes)),
                 median_bytes=float(np.median(sizes)), p95_bytes=float(np.percentile(sizes, 95)))
    for key, value in stats.items():
        if value != evaluation['feature_packets'][key]:
            raise ValueError('Frame sizes disagree with summary: ' + key)
    manifests = list(folder.glob('manifest_*.json'))
    if len(manifests) != 1:
        raise ValueError('Expected a single inference provenance manifest')
    provenance = json.loads(manifests[0].read_text(encoding='utf-8'))
    return ids, sizes, dict(directory=str(folder), statistics=stats,
        index_sha256=index_hash, manifest=provenance,
        evaluation=evaluation, offline_replay=replay)


def export(snn_dir, ann_dir, expected_frames):
    ids, snn, snn_report = read_run(snn_dir, expected_frames)
    ann_ids, ann, ann_report = read_run(ann_dir, expected_frames)
    if ids != ann_ids:
        raise ValueError('SNN and ANN frame IDs/order differ')
    for key in ('data_manifest_sha256', 'split_sha256'):
        if snn_report['manifest'][key] != ann_report['manifest'][key]:
            raise ValueError('SNN and ANN data provenance differs: ' + key)
    return dict(schema=1, passed=True, frames=len(ids), unit='byte',
        scope='One infrastructure-to-vehicle encoded feature packet per frame; includes codec headers and metadata, excludes network framing and retransmission',
        models={'snn': snn_report, 'ann': ann_report},
        per_frame=[dict(index=i, frame_id=fid, snn_bytes=s, ann_bytes=a)
                   for i, (fid, s, a) in enumerate(zip(ids, snn, ann))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snn-dir', required=True)
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-frames', type=int, default=1738)
    args = parser.parse_args()
    result = export(args.snn_dir, args.ann_dir, args.expected_frames)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({name: row['statistics'] for name, row in result['models'].items()}))

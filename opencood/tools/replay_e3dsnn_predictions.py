"""Verify a saved single-agent inference bundle and recompute AP without GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from opencood.utils import eval_utils, eval_3d_utils


def replay(folder):
    expected = json.loads((folder / 'evaluation.json').read_text())
    index_path = folder / 'predictions/index.json'
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == expected['predictions']['index_sha256']
    index = json.loads(index_path.read_text())
    assert index['schema'] == 1
    assert len(index['frames']) == expected['samples'] == expected['predictions']['frames']
    assert len({row['frame_id'] for row in index['frames']}) == len(index['frames'])
    assert len({row['file'] for row in index['frames']}) == len(index['frames'])
    stats = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (.3, .5, .7)}
    stats3d = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in stats}
    predicted, truth = 0, 0
    for position, row in enumerate(index['frames']):
        assert row['index'] == position
        path = folder / 'predictions' / row['file']
        assert path.resolve().parent == (folder / 'predictions').resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with np.load(str(path), allow_pickle=False) as data:
            assert str(data['frame_id'].item()) == row['frame_id']
            boxes, scores, gt = [torch.from_numpy(data[key].copy()) for key in ('pred_boxes', 'scores', 'gt_boxes')]
        assert boxes.shape == (row['predicted_boxes'], 8, 3)
        assert gt.shape == (row['gt_boxes'], 8, 3)
        assert scores.shape == (len(boxes),)
        assert all(torch.isfinite(value).all() for value in (boxes, scores, gt))
        for threshold in stats:
            eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
        if 'ap_3d' in expected:
            eval_3d_utils.calculate_tp_fp_3d(boxes, scores, gt, stats3d)
        predicted += len(boxes)
        truth += len(gt)
        if (position + 1) % 250 == 0:
            print(json.dumps({'replayed': position + 1, 'samples': len(index['frames'])}), flush=True)
    ap = {str(t): eval_utils.calculate_ap(stats, t)[0] for t in stats}
    assert ap == expected['ap'], (ap, expected['ap'])
    assert predicted == expected['predicted_boxes'] and truth == expected['gt_boxes']
    result = {'passed': True, 'samples': len(index['frames']), 'predicted_boxes': predicted,
              'gt_boxes': truth, 'ap': ap, 'exact_match': True, 'all_file_hashes_verified': True}
    if 'ap_3d' in expected:
        ap3d = {str(t): eval_utils.calculate_ap(stats3d, t)[0] for t in stats3d}
        assert ap3d == expected['ap_3d'], (ap3d, expected['ap_3d'])
        result['ap_3d'] = ap3d
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    report = replay(folder)
    (folder / 'offline_replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

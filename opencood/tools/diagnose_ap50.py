"""Range-stratified AP diagnosis for fixed predictions on the common CPU GT.

Answers where the AP50 gap comes from: the same immutable predictions are
re-scored in distance bins, so a model that is only weak on near/easy objects
shows a different profile from one that mislocalises everywhere. Read-only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from opencood.utils import eval_utils

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))


def read(path):
    return json.loads(Path(path).read_text())


def centre_distance(boxes):
    centres = boxes.mean(axis=1)
    return np.sqrt(centres[:, 0] ** 2 + centres[:, 1] ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    reference = Path(args.reference)
    evaluation = read(folder / 'evaluation.json')
    index = read(folder / 'predictions/index.json')['frames']
    refs = {row['frame_id']: row for row in read(reference / 'common_gt_manifest.json')['frames']}
    stats = {t: {b: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for b in range(len(BINS))}
             for t in (0.5, 0.7)}
    counts = {b: {'predictions': 0, 'gt': 0, 'frames': 0} for b in range(len(BINS))}
    for row in index:
        fid = row['frame_id']
        with np.load(str(folder / 'predictions' / row['file']), allow_pickle=False) as z:
            boxes = z['pred_boxes'].copy()
            scores = z['scores'].copy()
        gt = np.load(str(reference / 'common_gt' / (fid + '.npy')), allow_pickle=False)
        pd = centre_distance(boxes)
        gd = centre_distance(gt)
        for b, (lo, hi) in enumerate(BINS):
            mask = (pd >= lo) & (pd < hi)
            gmask = (gd >= lo) & (gd < hi)
            if mask.any() or gmask.any():
                counts[b]['frames'] += 1
            counts[b]['predictions'] += int(mask.sum())
            counts[b]['gt'] += int(gmask.sum())
            bin_stat = {k: stats[k][b] for k in stats}
            for t in stats:
                eval_utils.caluclate_tp_fp(torch.from_numpy(boxes[mask]), torch.from_numpy(scores[mask]),
                                           torch.from_numpy(gt[gmask]), bin_stat, t)
    result = {'samples': evaluation['samples'], 'gt_boxes': evaluation['gt_boxes'],
              'source_evaluation': str(folder / 'evaluation.json'),
              'bins': [], 'overall_ap': evaluation['ap']}
    for b, (lo, hi) in enumerate(BINS):
        row = {'range_m': [lo, None if hi > 1e8 else hi],
               'counts': counts[b]}
        for t in stats:
            row['AP%.1f' % t] = eval_utils.calculate_ap({k: stats[k][b] for k in stats}, t)[0]
        result['bins'].append(row)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

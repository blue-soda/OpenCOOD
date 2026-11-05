"""Range-stratified AP diagnosis for fixed predictions.

Answers where the AP50 gap comes from: the same immutable predictions are
re-scored in distance bins, so a model that is only weak on near/easy objects
shows a different profile from one that mislocalises everywhere. Read-only.

Two input formats:
  native : predictions/index.json + predictions/*.npz (pred_boxes, scores) and
           the common CPU GT under --reference.
  cosdh  : frames.jsonl + frames/*.npz (pred, score, gt) written by
           evaluate_cosdh; the GT is the evaluation's own GT.
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


def frames_native(folder, reference):
    for row in read(folder / 'predictions/index.json')['frames']:
        fid = row['frame_id']
        with np.load(str(folder / 'predictions' / row['file']), allow_pickle=False) as z:
            boxes, scores = z['pred_boxes'].copy(), z['scores'].copy()
        gt = np.load(str(reference / 'common_gt' / (fid + '.npy')), allow_pickle=False)
        yield boxes, scores, gt


def frames_cosdh(folder):
    rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        with np.load(str(folder / row['file']), allow_pickle=False) as z:
            yield z['pred'].copy(), z['score'].copy(), z['gt'].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--reference', default=None)
    parser.add_argument('--format', choices=('native', 'cosdh'), default='native')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    stats = {t: {b: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for b in range(len(BINS))}
             for t in (0.5, 0.7)}
    counts = {b: {'predictions': 0, 'gt': 0, 'frames': 0} for b in range(len(BINS))}
    source = frames_native(folder, Path(args.reference)) if args.format == 'native' else frames_cosdh(folder)
    for boxes, scores, gt in source:
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
    if args.format == 'native':
        evaluation = read(folder / 'evaluation.json')
        overall = evaluation['ap']
    else:
        summary = read(folder / 'summary.json')
        overall = summary['bev_ap']
    result = {'format': args.format, 'source_evaluation': str(folder),
              'bins': [], 'overall_ap': overall}
    for b, (lo, hi) in enumerate(BINS):
        row = {'range_m': [lo, None if hi > 1e8 else hi], 'counts': counts[b]}
        for t in stats:
            row['AP%.1f' % t] = eval_utils.calculate_ap({k: stats[k][b] for k in stats}, t)[0]
        result['bins'].append(row)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

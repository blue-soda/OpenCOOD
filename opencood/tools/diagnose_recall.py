"""Per-GT diagnosis: is a far GT missed, un-covered, or just low-scored?

For every ground-truth box (common cooperative GT) this walks the saved
predictions of one run and records, in fixed distance bins:

  n_gt            number of GT boxes in the bin
  covered         fraction with any prediction centre within 3 m
  recall@0.5/0.7  fraction matched at that IoU
  best_iou        mean of the best IoU over predictions
  best_score      mean score of the prediction with the best IoU
  low_score_match fraction of IoU>=0.5 matches whose score is below 0.2

Binning uses the GT centre distance, so a match cannot be split across bins.
Read-only: no training, no re-scoring, no threshold changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from opencood.utils import common_utils

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))
COVER_M = 3.0
LOW_SCORE = 0.2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--format', choices=('native', 'cosdh'), default='native')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    reference = Path(args.reference)
    if args.format == 'native':
        rows = json.loads((folder / 'predictions' / 'index.json').read_text())['frames']
        sources = [(row['frame_id'], str(folder / 'predictions' / row['file'])) for row in rows]
        pred_key, score_key = 'pred_boxes', 'scores'
    else:
        rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines() if line.strip()]
        sources = [(row['frame_id'], str(folder / row['file'])) for row in rows]
        pred_key, score_key = 'pred', 'score'
    bins = [{'n_gt': 0, 'covered': 0, 'm50': 0, 'm70': 0, 'best_iou': 0.0,
             'best_score': 0.0, 'low_score_match': 0} for _ in BINS]
    for frame_id, path in sources:
        with np.load(path, allow_pickle=False) as z:
            boxes = z[pred_key].copy()
            scores = z[score_key].copy()
        gt = np.load(str(reference / 'common_gt' / (frame_id + '.npy')), allow_pickle=False)
        if len(gt) == 0:
            continue
        pred_centres = boxes.mean(1) if len(boxes) else np.zeros((0, 3))
        gt_centres = gt.mean(1)
        pred_polys = list(common_utils.convert_format(boxes)) if len(boxes) else []
        gt_polys = list(common_utils.convert_format(gt))
        for g in range(len(gt)):
            radius = float(np.hypot(gt_centres[g, 0], gt_centres[g, 1]))
            for b, (lo, hi) in enumerate(BINS):
                if not (lo <= radius < hi):
                    continue
                entry = bins[b]
                entry['n_gt'] += 1
                if len(pred_centres):
                    nearest = float(np.min(np.hypot(pred_centres[:, 0] - gt_centres[g, 0],
                                                    pred_centres[:, 1] - gt_centres[g, 1])))
                    if nearest <= COVER_M:
                        entry['covered'] += 1
                if len(pred_polys):
                    ious = np.asarray([float(common_utils.compute_iou(p, [gt_polys[g]])[0]) for p in pred_polys])
                    best = float(np.max(ious))
                    best_idx = int(np.argmax(ious))
                    entry['best_iou'] += best
                    entry['best_score'] += float(scores[best_idx])
                    if best >= 0.5:
                        entry['m50'] += 1
                        if float(scores[best_idx]) < LOW_SCORE:
                            entry['low_score_match'] += 1
                    if best >= 0.7:
                        entry['m70'] += 1
                break
    out = {'source': str(folder), 'reference': str(reference), 'bins': []}
    for b, (lo, hi) in enumerate(BINS):
        e = bins[b]
        n = max(1, e['n_gt'])
        out['bins'].append({
            'range_m': [lo, None if hi > 1e8 else hi],
            'n_gt': e['n_gt'],
            'covered': round(e['covered'] / n, 4),
            'recall@0.5': round(e['m50'] / n, 4),
            'recall@0.7': round(e['m70'] / n, 4),
            'mean_best_iou': round(e['best_iou'] / n, 4),
            'mean_best_score': round(e['best_score'] / n, 4),
            'low_score_matches': e['low_score_match'],
        })
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

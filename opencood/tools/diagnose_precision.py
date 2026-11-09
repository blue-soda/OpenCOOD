"""Per-bucket precision diagnostic: candidates, true/false positives, threshold sweep.

Same fixed predictions as the recall diagnostic. A prediction is a true positive
if its BEV IoU with some ground-truth box reaches 0.5; buckets use the
prediction centre distance so a box is counted once. Score thresholds are
reported as a sweep, never used to retune anything.

Read-only: no training, no re-scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from opencood.utils import common_utils

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))
SCORE_THRESHOLDS = (0.3, 0.5, 0.7)


def load_source(folder, fmt):
    if fmt == 'native':
        rows = json.loads((folder / 'predictions' / 'index.json').read_text())['frames']
        return [(row['frame_id'], str(folder / 'predictions' / row['file']), 'pred_boxes', 'scores')
                for row in rows]
    rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines() if line.strip()]
    return [(row['frame_id'], str(folder / row['file']), 'pred', 'score') for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--format', choices=('native', 'cosdh'), default='native')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    reference = Path(args.reference)
    bins = [{'pred': 0, 'tp': 0} for _ in BINS]
    sweep = [{t: {'pred': 0, 'tp': 0} for t in SCORE_THRESHOLDS} for _ in BINS]
    for frame_id, path, pred_key, score_key in load_source(folder, args.format):
        with np.load(path, allow_pickle=False) as z:
            boxes = z[pred_key].copy()
            scores = z[score_key].copy()
        gt = np.load(str(reference / 'common_gt' / (frame_id + '.npy')), allow_pickle=False)
        if len(boxes) == 0:
            continue
        centres = boxes.mean(1)
        radius = np.hypot(centres[:, 0], centres[:, 1])
        gt_polys = list(common_utils.convert_format(gt)) if len(gt) else []
        for i in range(len(boxes)):
            poly = common_utils.convert_format(boxes[i:i + 1])[0]
            iou = float(np.max(common_utils.compute_iou(poly, gt_polys))) if gt_polys else 0.0
            is_tp = iou >= 0.5
            for b, (lo, hi) in enumerate(BINS):
                if lo <= radius[i] < hi:
                    bins[b]['pred'] += 1
                    bins[b]['tp'] += int(is_tp)
                    for t in SCORE_THRESHOLDS:
                        if scores[i] >= t:
                            sweep[b][t]['pred'] += 1
                            sweep[b][t]['tp'] += int(is_tp)
                    break
    out = {'source': str(folder), 'format': args.format, 'bins': []}
    for b, (lo, hi) in enumerate(BINS):
        entry = {'range_m': [lo, None if hi > 1e8 else hi],
                 'predictions': bins[b]['pred'], 'true_positives': bins[b]['tp'],
                 'precision@0.5': round(bins[b]['tp'] / max(1, bins[b]['pred']), 4)}
        for t in SCORE_THRESHOLDS:
            entry['precision_score>=%.1f' % t] = round(sweep[b][t]['tp'] / max(1, sweep[b][t]['pred']), 4)
            entry['pred_score>=%.1f' % t] = sweep[b][t]['pred']
        out['bins'].append(entry)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

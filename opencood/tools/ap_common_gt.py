"""Re-score saved predictions with the project evaluator on the common GT.

Two sources are supported:

  native : predictions/index.json + predictions/*.npz (pred_boxes, scores)
  cosdh  : frames.jsonl + frames/*.npz (pred, score)

The evaluator is ``opencood.utils.eval_utils`` with the same matching and AP
integration used for the native model, so every method is compared on one GT
and one protocol. Dataset-provided GT is ignored.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from opencood.utils import eval_utils


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
        sources = [(row['frame_id'], str(folder / 'predictions' / row['file']), 'pred_boxes', 'scores')
                   for row in rows]
    else:
        rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines() if line.strip()]
        sources = [(row['frame_id'], str(folder / row['file']), 'pred', 'score') for row in rows]
    stat = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (0.3, 0.5, 0.7)}
    for frame_id, path, pred_key, score_key in sources:
        with np.load(path, allow_pickle=False) as z:
            boxes = z[pred_key].copy()
            scores = z[score_key].copy()
        gt = np.load(str(reference / 'common_gt' / (frame_id + '.npy')), allow_pickle=False)
        for t in stat:
            eval_utils.caluclate_tp_fp(torch.from_numpy(boxes), torch.from_numpy(scores),
                                       torch.from_numpy(gt), stat, t)
    result = {
        'samples': len(sources),
        'gt_boxes': int(stat[0.5]['gt']),
        'metric': 'BEV AP with opencood eval_utils on the common cooperative GT',
        'ap': {str(t): eval_utils.calculate_ap(stat, t)[0] for t in stat},
        'source': str(folder),
        'format': args.format,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

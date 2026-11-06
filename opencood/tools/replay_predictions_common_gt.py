"""Score saved predictions on the common cooperative GT, ignoring dataset GT.

Single-vehicle training uses vehicle-local GT (22971 boxes), which is not
comparable to the cooperative GT (28622 boxes). This tool re-scores the saved
predictions of any run against the common cooperative reference so single and
cooperative models share one GT, split and AP protocol.
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
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.evaluation_dir)
    reference = Path(args.reference)
    index = json.loads((folder / 'predictions' / 'index.json').read_text())['frames']
    stat = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (.3, .5, .7)}
    for row in index:
        with np.load(str(folder / 'predictions' / row['file']), allow_pickle=False) as z:
            boxes = z['pred_boxes'].copy()
            scores = z['scores'].copy()
        gt = np.load(str(reference / 'common_gt' / (row['frame_id'] + '.npy')), allow_pickle=False)
        for threshold in stat:
            eval_utils.caluclate_tp_fp(torch.from_numpy(boxes), torch.from_numpy(scores),
                                       torch.from_numpy(gt), stat, threshold)
    result = {
        'samples': len(index),
        'gt_boxes': int(stat[0.5]['gt']),
        'metric': 'BEV AP on common cooperative GT; predictions unchanged; dataset GT ignored',
        'ap': {str(t): eval_utils.calculate_ap(stat, t)[0] for t in stat},
        'source_predictions': str(folder / 'predictions'),
        'reference': str(reference),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

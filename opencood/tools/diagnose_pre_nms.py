"""Pre-threshold candidate diagnostic: does a GT have a nearby high-score anchor?

The saved predictions are already thresholded and NMS-filtered, so they cannot
show whether a correct candidate existed before post-processing. This tool runs
the frozen model, takes the raw sigmoid scores, and for every ground-truth box
records the best score among anchors whose centre lies within ``--radius``
metres, per distance bucket. That separates two failure modes:

  * best nearby score below the operating threshold  -> classification / target
  * best nearby score high but no surviving box       -> NMS, max_num or decode

Read-only; one forward per frame, no training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))
SCORES = (0.05, 0.1, 0.2, 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    cfg['model']['args']['snapshot']['packet_directory'] = None
    device = torch.device('cuda:0')
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.to(device).eval()
    data = build_dataset(cfg, train=False)
    bins = [{'n_gt': 0, 'best': [], 'nearby': 0} for _ in BINS]
    frames = min(args.frames, len(data))
    with torch.no_grad():
        for index in range(frames):
            sample = data[index]
            ego = sample['ego']
            batch = train_utils.to_device(data.collate_batch_test([sample]), device)
            out = model(batch['ego'])
            psm = out['psm']
            prob = torch.sigmoid(psm.permute(0, 2, 3, 1)).reshape(-1).cpu().numpy()
            anchors = np.asarray(ego['anchor_box'])          # (H, W, A, 7)
            centres = anchors[..., :2].reshape(-1, 2)
            gt = np.asarray(ego['object_bbx_center'])
            mask = np.asarray(ego['object_bbx_mask']).reshape(-1).astype(bool)
            gt = gt[mask] if len(gt) == len(mask) else gt
            if len(gt) == 0:
                continue
            radius = np.hypot(gt[:, 0], gt[:, 1])
            for g in range(len(gt)):
                if radius[g] > 1e8:
                    continue
                near = np.hypot(centres[:, 0] - gt[g, 0], centres[:, 1] - gt[g, 1]) <= args.radius
                best = float(prob[near].max()) if near.any() else 0.0
                for b, (lo, hi) in enumerate(BINS):
                    if lo <= radius[g] < hi:
                        bins[b]['n_gt'] += 1
                        bins[b]['best'].append(best)
                        if best >= SCORES[2]:
                            bins[b]['nearby'] += 1
                        break
    result = {'frames': frames, 'radius_m': args.radius,
              'score_threshold_reference': 0.2, 'bins': []}
    for b, (lo, hi) in enumerate(BINS):
        e = bins[b]
        n = max(1, e['n_gt'])
        best = np.asarray(e['best']) if e['best'] else np.zeros(1)
        result['bins'].append({
            'range_m': [lo, None if hi > 1e8 else hi],
            'n_gt': e['n_gt'],
            'mean_best_nearby_score': round(float(best.mean()), 4),
            'frac_best_ge_0.05': round(float((best >= 0.05).mean()), 4),
            'frac_best_ge_0.1': round(float((best >= 0.1).mean()), 4),
            'frac_best_ge_0.2': round(float((best >= 0.2).mean()), 4),
            'frac_best_ge_0.5': round(float((best >= 0.5).mean()), 4),
        })
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

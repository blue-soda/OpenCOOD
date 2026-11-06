"""Count positive, negative and ignored anchors, by distance, before training.

The label generator marks an anchor positive when its best IoU reaches
pos_threshold, negative when every IoU is below neg_threshold, and leaves the
remaining mid-IoU anchors in neither set (ignored). The current classification
loss penalises every non-positive anchor, ignored ones included. This tool
measures the size and distance distribution of each group so the fix can be
sized before spending GPU time.

Read-only: no training, no GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    cfg['model']['args']['snapshot']['packet_directory'] = None
    data = build_dataset(cfg, train=True)
    anchor_args = cfg['postprocess']['anchor_args']
    stride = anchor_args['feature_stride']
    x0, y0 = anchor_args['cav_lidar_range'][0], anchor_args['cav_lidar_range'][1]
    bins = [{'pos': 0, 'neg': 0, 'ignore': 0} for _ in BINS]
    frames = min(args.frames, len(data))
    total = {'pos': 0, 'neg': 0, 'ignore': 0, 'anchors': 0}
    for index in range(frames):
        label = data[index]['ego']['label_dict']
        pos = np.asarray(label['pos_equal_one']) > 0
        neg = np.asarray(label['neg_equal_one']) > 0
        h, w, anchors = pos.shape
        x1, y1 = anchor_args['cav_lidar_range'][3], anchor_args['cav_lidar_range'][4]
        xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
        ys = y0 + (np.arange(h) + 0.5) * (y1 - y0) / h
        gx, gy = np.meshgrid(xs, ys)
        radius = np.sqrt(gx ** 2 + gy ** 2)[..., None].repeat(anchors, axis=-1)
        ignore = ~(pos | neg)
        total['pos'] += int(pos.sum())
        total['neg'] += int(neg.sum())
        total['ignore'] += int(ignore.sum())
        total['anchors'] += int(pos.size)
        for b, (lo, hi) in enumerate(BINS):
            m = (radius >= lo) & (radius < hi)
            bins[b]['pos'] += int((pos & m).sum())
            bins[b]['neg'] += int((neg & m).sum())
            bins[b]['ignore'] += int((ignore & m).sum())
    result = {'frames': frames, 'totals': total,
              'ignore_fraction': round(total['ignore'] / max(1, total['anchors']), 6),
              'bins': []}
    for b, (lo, hi) in enumerate(BINS):
        e = bins[b]
        s = e['pos'] + e['neg'] + e['ignore']
        result['bins'].append({'range_m': [lo, None if hi > 1e8 else hi],
                               'pos': e['pos'], 'neg': e['neg'], 'ignore': e['ignore'],
                               'ignore_fraction': round(e['ignore'] / max(1, s), 6)})
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

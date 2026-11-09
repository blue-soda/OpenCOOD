"""Stage-wise candidate diagnostic: where do ground-truth boxes disappear?

For every frame this tool runs the frozen model, decodes all anchors, and walks
the post-processing pipeline in the same order as the evaluator:

  raw -> threshold(0.2) -> rotated NMS(0.15) -> top-K(100)

At each stage it reports one-to-one matches against the ground truth at IoU 0.5
and 0.7, with denominators split by range. It also records, for every unmatched
GT, whether a pre-threshold candidate (score >= 0.2, anchor centre <= 2 m)
existed, so a GT lost at NMS/top-K is distinguishable from one that had no
candidate at all.

Read-only; one forward per frame, no training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils

BINS = ((0.0, 30.0), (30.0, 50.0), (50.0, 80.0), (80.0, 1e9))
IOUS = (0.5, 0.7)


def bev_corners(boxes):
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    corners = np.stack([
        np.stack([x + c * dx - s * dy, y + s * dx + c * dy], -1),
        np.stack([x + c * dx + s * dy, y + s * dx - c * dy], -1),
        np.stack([x - c * dx + s * dy, y - s * dx - c * dy], -1),
        np.stack([x - c * dx - s * dy, y - s * dx + c * dy], -1),
    ], 1)
    return corners


def iou_matrix(boxes, gts):
    if len(boxes) == 0 or len(gts) == 0:
        return np.zeros((len(gts), len(boxes)))
    bc = bev_corners(boxes)
    gc = bev_corners(gts)
    out = np.zeros((len(gts), len(boxes)))
    gpolys = [Polygon(g).buffer(0) for g in gc]
    for i, gp in enumerate(gpolys):
        for j, bcj in enumerate(bc):
            bp = Polygon(bcj).buffer(0)
            inter = gp.intersection(bp).area
            if inter <= 0:
                continue
            out[i, j] = inter / (gp.area + bp.area - inter)
    return out


def nms_bev(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    polys = [Polygon(c).buffer(0) for c in bev_corners(boxes)] if len(boxes) else []
    for j in order:
        ok = True
        for k in keep:
            inter = polys[j].intersection(polys[k]).area
            if inter <= 0:
                continue
            if inter / (polys[j].area + polys[k].area - inter) > thr:
                ok = False
                break
        if ok:
            keep.append(j)
    return np.asarray(keep, dtype=np.int64)


def match_counts(boxes, scores, gts, iou_thr):
    """Greedy one-to-one by score; returns (tp, fp, matched_gt_indices)."""
    order = np.argsort(-scores)
    used = np.zeros(len(gts), dtype=bool)
    tp = 0
    fp = 0
    matched = []
    ious = np.zeros((len(gts), len(boxes)))
    if len(boxes):
        near = np.zeros(len(boxes), dtype=bool)
        for i in range(len(gts)):
            d = np.hypot(boxes[:, 0] - gts[i, 0], boxes[:, 1] - gts[i, 1])
            near |= d <= 6.0
        if near.any():
            ious[:, near] = iou_matrix(boxes[near], gts)
    for j in order:
        best, best_iou = -1, iou_thr
        for i in range(len(gts)):
            if used[i] or ious[i, j] < best_iou:
                continue
            best, best_iou = i, ious[i, j]
        if best >= 0:
            used[best] = True
            tp += 1
            matched.append(best)
        else:
            fp += 1
    return tp, fp, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    cfg['model']['args']['snapshot']['packet_directory'] = None
    device = torch.device('cuda:0')
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.to(device).eval()
    data = build_dataset(cfg, train=False)
    post = data.post_processor
    params = post.params
    thr = float(params['target_args']['score_threshold'])
    nms_thr = float(params['nms_thresh'])
    max_num = int(params['max_num'])
    bins = [{'n_gt': 0, 'stages': {name: {str(t): [0, 0] for t in IOUS} for name in ('pre', 'thr', 'nms', 'topk')},
             'no_candidate': 0} for _ in BINS]
    frames = min(args.frames, len(data))
    with torch.no_grad():
        for index in range(frames):
            sample = data[index]
            batch = train_utils.to_device(data.collate_batch_test([sample]), device)
            out = model(batch['ego'])
            prob = torch.sigmoid(out['psm'].permute(0, 2, 3, 1)).reshape(-1)
            anchors = torch.from_numpy(np.asarray(sample['ego']['anchor_box'])).float().to(out['rm'].device)
            boxes = VoxelPostprocessor.delta_to_boxes3d(out['rm'], anchors)[0]
            boxes_np = boxes.detach().cpu().numpy()
            scores = prob.detach().cpu().numpy()
            gt = np.asarray(sample['ego']['object_bbx_center']).reshape(-1, 7)
            gmask = np.asarray(sample['ego']['object_bbx_mask']).reshape(-1).astype(bool)
            gt = gt[gmask]
            if len(gt) == 0:
                continue
            radius = np.hypot(gt[:, 0], gt[:, 1])
            stages = {}
            stages['pre'] = scores >= 0.1
            stages['thr'] = scores >= thr
            if stages['thr'].any():
                keep = nms_bev(boxes_np[stages['thr']], scores[stages['thr']], nms_thr)
                idx = np.flatnonzero(stages['thr'])[np.asarray(keep)]
                mask_nms = np.zeros(len(scores), dtype=bool)
                mask_nms[idx] = True
            else:
                mask_nms = np.zeros(len(scores), dtype=bool)
            stages['nms'] = mask_nms
            order = np.argsort(-scores)
            top = np.zeros(len(scores), dtype=bool)
            top[np.flatnonzero(mask_nms)[np.argsort(-scores[mask_nms])][:max_num]] = True
            stages['topk'] = top
            for b, (lo, hi) in enumerate(BINS):
                m = (radius >= lo) & (radius < hi)
                sub = np.flatnonzero(m)
                if len(sub) == 0:
                    continue
                bins[b]['n_gt'] += len(sub)
                for name, mask in stages.items():
                    boxes_s = boxes_np[mask]
                    scores_s = scores[mask]
                    for t in IOUS:
                        tp, _, _ = match_counts(boxes_s, scores_s, gt[sub], t)
                        bins[b]['stages'][name][str(t)][0] += tp
                        bins[b]['stages'][name][str(t)][1] += len(sub)
                # GT without any thresholded candidate within 2 m
                centres = boxes_np[stages['thr'], :2]
                for i in sub:
                    near = np.hypot(centres[:, 0] - gt[i, 0], centres[:, 1] - gt[i, 1]) <= 2.0
                    if not near.any():
                        bins[b]['no_candidate'] += 1
    result = {'frames': frames, 'score_threshold': thr, 'nms_threshold': nms_thr,
              'max_num': max_num, 'bins': []}
    for b, (lo, hi) in enumerate(BINS):
        e = bins[b]
        n = max(1, e['n_gt'])
        entry = {'range_m': [lo, None if hi > 1e8 else hi], 'n_gt': e['n_gt'],
                 'no_candidate': e['no_candidate'],
                 'recall': {}}
        for name in ('pre', 'thr', 'nms', 'topk'):
            entry['recall'][name] = {str(t): round(e['stages'][name][str(t)][0] / n, 4) for t in IOUS}
        result['bins'].append(entry)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

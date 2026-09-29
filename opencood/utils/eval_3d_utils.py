"""Upright, yaw-rotated 3D box AP with the existing OpenCOOD AP convention.

Input corners must follow OpenCOOD's (N, 8, 3) convention: the first four
corners form one horizontal face. This is not KITTI R11/R40 evaluation.
"""
import numpy as np
from shapely.geometry import Polygon


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _geometry(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3) or not np.isfinite(boxes).all():
        raise ValueError('Expected finite (N, 8, 3) corners')
    if len(boxes) and (not np.allclose(boxes[:, :4, 2], boxes[:, :1, 2], atol=1e-4, rtol=0)
                      or not np.allclose(boxes[:, 4:, 2], boxes[:, 4:5, 2], atol=1e-4, rtol=0)
                      or not np.allclose(boxes[:, :4, :2], boxes[:, 4:, :2], atol=1e-4, rtol=0)):
        raise ValueError('3D evaluator requires upright boxes, without roll/pitch')
    polygons = [Polygon(box[:4, :2]) for box in boxes]
    lower, upper = boxes[:, :, 2].min(axis=1), boxes[:, :, 2].max(axis=1)
    areas = np.asarray([polygon.area for polygon in polygons])
    if any(not polygon.is_valid for polygon in polygons) or np.any(areas <= 0) or np.any(upper <= lower):
        raise ValueError('Invalid or degenerate box geometry')
    return polygons, lower, upper, areas * (upper - lower)


def pairwise_iou3d(det_boxes, gt_boxes):
    """Exact planar polygon intersection times vertical overlap / volume union."""
    det, det_low, det_high, det_vol = _geometry(as_numpy(det_boxes))
    gt, gt_low, gt_high, gt_vol = _geometry(as_numpy(gt_boxes))
    ious = np.zeros((len(det), len(gt)), dtype=np.float64)
    for i, polygon in enumerate(det):
        heights = np.maximum(0., np.minimum(det_high[i], gt_high) - np.maximum(det_low[i], gt_low))
        for j in np.flatnonzero(heights > 0):
            intersection = polygon.intersection(gt[j]).area * heights[j]
            ious[i, j] = intersection / (det_vol[i] + gt_vol[j] - intersection)
    return ious


def calculate_tp_fp_3d(det_boxes, det_score, gt_boxes, result_stat):
    """Accumulate all thresholds; greedy score order and one match per GT.

    Use eval_utils.calculate_ap for the unchanged global confidence sorting
    and VOC2010 precision-envelope integral.
    """
    gt_boxes = as_numpy(gt_boxes)
    if det_boxes is None:
        for stat in result_stat.values():
            stat['gt'] += len(gt_boxes)
        return
    scores = as_numpy(det_score)
    boxes = as_numpy(det_boxes)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite confidence score per prediction')
    order = np.argsort(-scores)
    ious = pairwise_iou3d(boxes, gt_boxes)
    for threshold, stat in result_stat.items():
        unmatched = list(range(len(gt_boxes)))
        for index in order:
            candidates = ious[index, unmatched]
            matched = bool(len(unmatched) and candidates.max() >= threshold)
            stat['tp'].append(int(matched))
            stat['fp'].append(int(not matched))
            if matched:
                unmatched.pop(int(candidates.argmax()))
        stat['score'].extend(scores[order].tolist())
        stat['gt'] += len(gt_boxes)

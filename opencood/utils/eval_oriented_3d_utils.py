"""Volume IoU for OpenCOOD cuboids that may have roll and pitch.

Corners 0..3 and 4..7 form the two corresponding faces. Construct the
intersection from corners inside both boxes and intersections of each box's
edges with the other's faces. Convexity makes these sufficient.
"""
import numpy as np
from scipy.spatial import ConvexHull

from opencood.utils.eval_3d_utils import as_numpy


_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                   (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])


def _clipped_vertices(corners, opposite, both):
    start = corners[_EDGES[:, 0]]
    direction = corners[_EDGES[:, 1]] - start
    distance = start @ opposite[:, :3].T + opposite[:, 3]
    slope = direction @ opposite[:, :3].T
    fraction = np.divide(-distance, slope, out=np.full_like(distance, np.inf),
                         where=np.abs(slope) > 1e-12)
    edge, plane = np.where((fraction >= 0) & (fraction <= 1))
    crossings = start[edge] + fraction[edge, plane, None] * direction[edge]
    points = np.concatenate([corners, crossings])
    return points[(points @ both[:, :3].T + both[:, 3] <= 1e-6).all(axis=1)]


def _corners(boxes):
    boxes = np.asarray(as_numpy(boxes), dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3) or not np.isfinite(boxes).all():
        raise ValueError('Expected finite (N, 8, 3) OpenCOOD corners')
    return boxes


def pairwise_oriented_iou3d(det_boxes, gt_boxes):
    a, b = _corners(det_boxes), _corners(gt_boxes)
    iou = np.zeros((len(a), len(b)))
    if not len(a) or not len(b):
        return iou
    candidates = ((a.min(1)[:, None, :] < b.max(1)[None, :, :]).all(2)
                  & (a.max(1)[:, None, :] > b.min(1)[None, :, :]).all(2))
    det_hulls, gt_hulls = {}, {}
    for i, j in zip(*np.where(candidates)):
        if i not in det_hulls:
            det_hulls[i] = ConvexHull(a[i])
        if j not in gt_hulls:
            gt_hulls[j] = ConvexHull(b[j])
        ha, hb = det_hulls[i], gt_hulls[j]
        both = np.concatenate([ha.equations, hb.equations])
        vertices = np.unique(np.concatenate([
            _clipped_vertices(a[i], hb.equations, both),
            _clipped_vertices(b[j], ha.equations, both)]), axis=0)
        if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices[0], tol=1e-7) < 3:
            continue
        intersection = ConvexHull(vertices).volume
        if intersection > min(ha.volume, hb.volume) * (1 + 1e-5):
            raise ValueError('Intersection larger than a cuboid; check corner order')
        iou[i, j] = min(intersection / (ha.volume + hb.volume - intersection), 1.)
    return iou

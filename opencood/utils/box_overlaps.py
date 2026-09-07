"""Pure Python fallback for the optional Cython box_overlaps extension."""

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """Compute IoU overlaps between two sets of axis-aligned boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Shape ``(N, 4)`` with ``[x1, y1, x2, y2]`` boxes.
    query_boxes : np.ndarray
        Shape ``(K, 4)`` with ``[x1, y1, x2, y2]`` boxes.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    if boxes.size == 0 or query_boxes.size == 0:
        return overlaps

    box_areas = (
        (boxes[:, 2] - boxes[:, 0] + 1.0)
        * (boxes[:, 3] - boxes[:, 1] + 1.0)
    )
    query_areas = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1.0)
        * (query_boxes[:, 3] - query_boxes[:, 1] + 1.0)
    )

    for k in range(query_boxes.shape[0]):
        iw = (
            np.minimum(boxes[:, 2], query_boxes[k, 2])
            - np.maximum(boxes[:, 0], query_boxes[k, 0])
            + 1.0
        )
        ih = (
            np.minimum(boxes[:, 3], query_boxes[k, 3])
            - np.maximum(boxes[:, 1], query_boxes[k, 1])
            + 1.0
        )
        valid = (iw > 0) & (ih > 0)
        inter = iw[valid] * ih[valid]
        union = box_areas[valid] + query_areas[k] - inter
        overlaps[valid, k] = inter / union
    return overlaps


def bbox_intersections(boxes, query_boxes):
    """Compute query-box coverage ratio by boxes."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    intersec = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    if boxes.size == 0 or query_boxes.size == 0:
        return intersec

    query_areas = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1.0)
        * (query_boxes[:, 3] - query_boxes[:, 1] + 1.0)
    )
    for k in range(query_boxes.shape[0]):
        iw = (
            np.minimum(boxes[:, 2], query_boxes[k, 2])
            - np.maximum(boxes[:, 0], query_boxes[k, 0])
            + 1.0
        )
        ih = (
            np.minimum(boxes[:, 3], query_boxes[k, 3])
            - np.maximum(boxes[:, 1], query_boxes[k, 1])
            + 1.0
        )
        valid = (iw > 0) & (ih > 0)
        intersec[valid, k] = (iw[valid] * ih[valid]) / query_areas[k]
    return intersec

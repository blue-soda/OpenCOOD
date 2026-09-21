"""Optional GPU ROI post-processing for CoBEVFlow.

This module is deliberately separate from the original post-processor.  It
keeps the original decode, projection, score threshold, top-k, geometry
filters, NMS threshold and bandwidth filtering.  The two operations that are
replaced are the CPU-only ``corner_to_center_torch`` conversion and the
Shapely-based rotated NMS.

The rotated NMS below computes the same rectangle intersection-over-union
definition as ``box_utils.nms_rotated``.  It uses torch tensors, so the input
does not have to leave the CUDA device.  ``cobevflow_fast_single_post_process``
is a copy of the existing single-frame path with those two operations
injected; the original implementation remains untouched.
"""

from collections import OrderedDict
import math

import numpy as np
import torch

from opencood.utils import box_utils
from opencood.data_utils.post_processor.voxel_postprocessor import limit_period

try:
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import (
        nms_gpu as _cuda_rotated_nms,
    )
except Exception:  # pragma: no cover - optional compiled extension
    _cuda_rotated_nms = None


def corner_to_center_torch_gpu(corner3d, order='lwh'):
    """Torch equivalent of ``box_utils.corner_to_center_torch``.

    The indexing and averaging follow the original NumPy implementation so
    that this function is only a device-placement optimization.
    """
    xyz = corner3d[:, [0, 3, 5, 6], :].mean(dim=1)
    h = (corner3d[:, 4:, 2] - corner3d[:, :4, 2]).mean(dim=1).abs().unsqueeze(1)

    def _edge_length(i, j):
        return torch.linalg.vector_norm(corner3d[:, i, :2] - corner3d[:, j, :2],
                                        dim=1, keepdim=True)

    l = (_edge_length(0, 3) + _edge_length(2, 1) +
         _edge_length(4, 7) + _edge_length(5, 6)) / 4.0
    w = (_edge_length(0, 1) + _edge_length(2, 3) +
         _edge_length(4, 5) + _edge_length(6, 7)) / 4.0

    theta = (
        torch.atan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                    corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
        torch.atan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                    corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
        torch.atan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                    corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
        torch.atan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                    corner3d[:, 4, 0] - corner3d[:, 7, 0])
    ).unsqueeze(1) / 4.0

    if order == 'lwh':
        return torch.cat((xyz, l, w, h, theta), dim=1).reshape(-1, 7)
    if order == 'hwl':
        return torch.cat((xyz, h, w, l, theta), dim=1).reshape(-1, 7)
    raise ValueError('Unknown order: {}'.format(order))


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _signed_polygon_area(poly):
    return 0.5 * (
        poly[..., 0] * torch.roll(poly[..., 1], shifts=-1, dims=-1) -
        poly[..., 1] * torch.roll(poly[..., 0], shifts=-1, dims=-1)
    ).sum(dim=-1)


def _points_inside_convex(poly, points):
    """Return [M,K] for points inside a 4-corner convex polygon."""
    edges = torch.roll(poly, shifts=-1, dims=0) - poly
    rel = points.unsqueeze(1) - poly.unsqueeze(0)
    cross = _cross2d(edges.unsqueeze(0), rel)
    orientation = torch.sign(_signed_polygon_area(poly)).clamp(min=-1.0, max=1.0)
    return (cross * orientation >= -1e-5).all(dim=1)


def _rectangle_intersection_iou(rect, others):
    """IoU between one rectangle and a batch of rectangles on the same device.

    The intersection polygon is formed by the two sets of rectangle corners
    and all pairwise segment intersections, then its vertices are angle-sorted.
    This is the convex-polygon equivalent of the Shapely operation used by
    the original implementation.
    """
    device = others.device
    dtype = others.dtype
    count = others.shape[0]
    if count == 0:
        return others.new_empty((0,))

    rect_area = _signed_polygon_area(rect).abs()
    other_area = _signed_polygon_area(others).abs()

    rect_points = rect.unsqueeze(0).expand(count, -1, -1)
    candidates = [rect_points, others]
    valid = [
        _points_inside_convex(others[0], rect_points) if False else None
    ]

    # Test corners of A against each B and corners of B against A.
    edges_b = torch.roll(others, shifts=-1, dims=1) - others
    rel_a = rect.unsqueeze(0).unsqueeze(2) - others.unsqueeze(1)
    cross_a = _cross2d(edges_b.unsqueeze(1), rel_a)
    orient_b = torch.sign(_signed_polygon_area(others)).clamp(min=-1.0, max=1.0)
    inside_a = (cross_a * orient_b.view(-1, 1, 1) >= -1e-5).all(dim=2)
    inside_b = _points_inside_convex(rect, others)

    # Pairwise segment intersections, A edge x B edge.
    a0 = rect
    a1 = torch.roll(rect, shifts=-1, dims=0)
    b0 = others
    b1 = torch.roll(others, shifts=-1, dims=1)
    r = (a1 - a0).unsqueeze(0).unsqueeze(2)       # [1,4,1,2]
    s = (b1 - b0).unsqueeze(1)                    # [M,1,4,2]
    qp = b0.unsqueeze(1) - a0.unsqueeze(0).unsqueeze(2)
    denom = _cross2d(r, s)
    denom_safe = torch.where(denom.abs() > 1e-8, denom,
                             torch.ones_like(denom))
    t = _cross2d(qp, s) / denom_safe
    u = _cross2d(qp, r) / denom_safe
    valid_intersection = (
        (denom.abs() > 1e-8) & (t >= 0.0) & (t <= 1.0) &
        (u >= 0.0) & (u <= 1.0)
    )
    intersections = a0.unsqueeze(0).unsqueeze(2) + t.unsqueeze(-1) * r
    intersections = intersections.reshape(count, 16, 2)
    valid_intersection = valid_intersection.reshape(count, 16)

    points = torch.cat((rect_points, others, intersections), dim=1)
    valid = torch.cat((inside_a, inside_b, valid_intersection), dim=1)
    valid_count = valid.sum(dim=1)
    safe_count = valid_count.clamp(min=1)
    points_for_mean = torch.where(valid.unsqueeze(-1), points,
                                  torch.zeros_like(points))
    center = points_for_mean.sum(dim=1) / safe_count.unsqueeze(1).to(dtype)
    angles = torch.atan2(points[..., 1] - center[:, None, 1],
                         points[..., 0] - center[:, None, 0])
    angles = torch.where(valid, angles, torch.full_like(angles, 4.0 * math.pi))
    order = torch.argsort(angles, dim=1)
    sorted_points = torch.gather(points, 1, order.unsqueeze(-1).expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)
    next_points = torch.roll(sorted_points, shifts=-1, dims=1)
    next_valid = torch.roll(sorted_valid, shifts=-1, dims=1)
    edge_valid = sorted_valid & next_valid
    signed_area = 0.5 * _cross2d(sorted_points, next_points)
    intersection_area = (signed_area * edge_valid.to(dtype)).sum(dim=1).abs()
    union = rect_area + other_area - intersection_area
    return torch.where(union > 1e-8, intersection_area / union,
                       torch.zeros_like(union))


def rotated_nms_gpu(boxes, scores, threshold, top=1000):
    """GPU rotated NMS with the same top-k and IoU threshold as the baseline."""
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # The repository's compiled PCDet kernel computes oriented BEV IoU on the
    # same rectangle geometry as the original Shapely path.  The postprocessor
    # uses ``hwl`` boxes, while the kernel expects ``lwh`` dimensions.
    if _cuda_rotated_nms is not None and boxes.is_cuda:
        centers_hwl = corner_to_center_torch_gpu(boxes, order='hwl')
        centers_lwh = centers_hwl[:, [0, 1, 2, 5, 4, 3, 6]].contiguous()
        keep, _ = _cuda_rotated_nms(
            centers_lwh, scores, threshold, pre_maxsize=top)
        return keep.to(device=boxes.device, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    order = order[:top]
    rects = boxes[order, :4, :2].contiguous()
    remaining = torch.arange(order.shape[0], device=boxes.device)
    kept = []
    while remaining.numel() > 0:
        current = remaining[0]
        kept.append(order[current])
        if remaining.numel() == 1:
            break
        iou = _rectangle_intersection_iou(rects[current], rects[remaining[1:]])
        remaining = remaining[1:][iou <= threshold]
    if not kept:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(kept).to(dtype=torch.long)


def _empty(device, shape):
    return torch.empty(shape, device=device)


def cobevflow_fast_single_post_process(post_processor, m_single,
                                       trans_mat_pastk_2_past0,
                                       past_time_diff, anchor_box,
                                       num_sweeps=2, num_roi_thres=-1):
    """GPU-preserving equivalent of ``VoxelPostprocessor.single_post_process``."""
    post_processor.k = num_sweeps
    post_processor.num_roi_thres = num_roi_thres
    psm_single = m_single['psm_single']
    rm_single = m_single['rm_single']
    has_uncertainty = ('predict_unc_cls' in m_single and
                       'predict_unc_reg' in m_single)
    use_dir_flag = 'dm_single' in m_single
    dm_single = m_single.get('dm_single')
    box_results = OrderedDict()
    box_results['past_k_time_diff'] = past_time_diff
    transformation_matrix = trans_mat_pastk_2_past0.to(torch.float32)

    prob = torch.sigmoid(psm_single.permute(0, 2, 3, 1)).reshape(num_sweeps, -1)
    reg = rm_single
    batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)
    mask = torch.gt(prob, post_processor.params['target_args']['score_threshold'])
    mask = mask.view(num_sweeps, -1)
    mask_reg = mask.unsqueeze(2).expand(-1, -1, 7)

    if has_uncertainty:
        predict_unc_cls = m_single['predict_unc_cls']
        predict_unc_reg = m_single['predict_unc_reg']
        cls_data_unc = predict_unc_cls[:, :post_processor.anchor_num].permute(0, 2, 3, 1).reshape(num_sweeps, -1)
        cls_model_unc = predict_unc_cls[:, post_processor.anchor_num:].permute(0, 2, 3, 1).reshape(num_sweeps, -1)
        reg_data_unc = predict_unc_reg[:, :post_processor.anchor_num].permute(0, 2, 3, 1).reshape(num_sweeps, -1)
        reg_model_unc = predict_unc_reg[:, post_processor.anchor_num:].permute(0, 2, 3, 1).reshape(num_sweeps, -1)

    if use_dir_flag:
        dir_offset = post_processor.params['dir_args']['dir_offset']
        num_bins = post_processor.params['dir_args']['num_bins']
        dm = dm_single

    for i in range(num_sweeps):
        box_results[i] = OrderedDict()
        unit_trans_mat = transformation_matrix[i]
        boxes3d = torch.masked_select(batch_box3d[i], mask_reg[i]).view(-1, 7)
        scores = torch.masked_select(prob[i], mask[i])
        pre_nms_topk_idx = None
        if has_uncertainty:
            u_cls_data = torch.masked_select(cls_data_unc[i], mask[i])
            u_cls_model = torch.masked_select(cls_model_unc[i], mask[i])
            u_reg_data = torch.masked_select(reg_data_unc[i], mask[i])
            u_reg_model = torch.masked_select(reg_model_unc[i], mask[i])

        roi_pre_nms_topk = post_processor.params['target_args'].get('roi_pre_nms_topk', -1)
        if roi_pre_nms_topk is not None and roi_pre_nms_topk > 0 and scores.shape[0] > roi_pre_nms_topk:
            topk_scores, topk_idx = torch.topk(scores, int(roi_pre_nms_topk), largest=True, sorted=False)
            boxes3d, scores = boxes3d[topk_idx], topk_scores
            pre_nms_topk_idx = topk_idx
            if has_uncertainty:
                u_cls_data, u_cls_model = u_cls_data[topk_idx], u_cls_model[topk_idx]
                u_reg_data, u_reg_model = u_reg_data[topk_idx], u_reg_model[topk_idx]

        if use_dir_flag and len(boxes3d) != 0:
            dir_cls_preds = dm[i:i + 1].permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
            dir_cls_preds = dir_cls_preds[mask[i:i + 1]]
            if pre_nms_topk_idx is not None:
                dir_cls_preds = dir_cls_preds[pre_nms_topk_idx]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = 2 * np.pi / num_bins
            dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
            boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)

        if len(boxes3d) != 0:
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=post_processor.params['order'])
            projected_boxes3d = box_utils.project_box3d(boxes3d_corner, unit_trans_mat)
            keep_index = box_utils.remove_large_pred_bbx(projected_boxes3d)
            keep_index = torch.logical_and(keep_index, box_utils.remove_bbx_abnormal_z(projected_boxes3d))
            projected_boxes3d, scores = projected_boxes3d[keep_index], scores[keep_index]
            if has_uncertainty:
                u_cls_data, u_cls_model = u_cls_data[keep_index], u_cls_model[keep_index]
                u_reg_data, u_reg_model = u_reg_data[keep_index], u_reg_model[keep_index]

            keep_index = rotated_nms_gpu(projected_boxes3d, scores,
                                         post_processor.params['nms_thresh'])
            pred_box3d_tensor = projected_boxes3d[keep_index]
            scores = scores[keep_index]
            if has_uncertainty:
                u_cls_data, u_cls_model = u_cls_data[keep_index], u_cls_model[keep_index]
                u_reg_data, u_reg_model = u_reg_data[keep_index], u_reg_model[keep_index]

            range_mask = box_utils.get_mask_for_boxes_within_range_torch(
                pred_box3d_tensor, post_processor.params['gt_range'])
            pred_box_3dcorner_tensor = pred_box3d_tensor[range_mask]
            scores = scores[range_mask]
            pred_box_center_tensor = corner_to_center_torch_gpu(
                pred_box_3dcorner_tensor, post_processor.params['order'])
            if post_processor.num_roi_thres != -1:
                top_idx = torch.argsort(scores, descending=True)[:post_processor.num_roi_thres]
                pred_box_3dcorner_tensor = pred_box_3dcorner_tensor[top_idx]
                pred_box_center_tensor = pred_box_center_tensor[top_idx]
                scores = scores[top_idx]
            box_results[i].update({
                'pred_box_3dcorner_tensor': pred_box_3dcorner_tensor,
                'pred_box_center_tensor': pred_box_center_tensor,
                'scores': scores,
            })
            if has_uncertainty:
                box_results[i].update({
                    'u_cls_data': u_cls_data[range_mask],
                    'u_cls_model': u_cls_model[range_mask],
                    'u_reg_data': u_reg_data[range_mask],
                    'u_reg_model': u_reg_model[range_mask],
                })
        else:
            box_results[i].update({
                'pred_box_3dcorner_tensor': _empty(scores.device, (0, 8, 3)),
                'pred_box_center_tensor': _empty(scores.device, (0, 7)),
                'scores': _empty(scores.device, (0,)),
            })
            if has_uncertainty:
                for key in ('u_cls_data', 'u_cls_model', 'u_reg_data', 'u_reg_model'):
                    box_results[i][key] = _empty(scores.device, (0,))
    return box_results

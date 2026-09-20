"""Metric BEV resampling. Transforms map source metric points to target."""

import torch
import torch.nn.functional as F


def warp_metric_bev(source, source_to_target, lidar_range):
    n, _, h, w = source.shape
    transform = source_to_target.to(source).reshape(n, 4, 4)
    xmin, ymin, _, xmax, ymax, _ = lidar_range
    xs = xmin + (torch.arange(w, device=source.device, dtype=source.dtype) + .5) * ((xmax - xmin) / w)
    ys = ymin + (torch.arange(h, device=source.device, dtype=source.dtype) + .5) * ((ymax - ymin) / h)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    target = torch.stack((xx, yy, torch.zeros_like(xx), torch.ones_like(xx)), -1)
    # grid_sample is a backwards sampler; invert the metric forward transform.
    points = torch.einsum('nij,hwj->nhwi', torch.linalg.inv(transform), target)
    grid = torch.stack((2 * (points[..., 0] - xmin) / (xmax - xmin) - 1,
                        2 * (points[..., 1] - ymin) / (ymax - ymin) - 1), -1)
    warped = F.grid_sample(source, grid, align_corners=False, padding_mode='zeros')
    coverage = F.grid_sample(torch.ones_like(source[:, :1]), grid,
                             align_corners=False, padding_mode='zeros')
    return warped, coverage.clamp(0, 1)

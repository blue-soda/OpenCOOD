"""Geometry and conservative registration for timestamped sparse messages."""
import torch
from scipy.optimize import linear_sum_assignment


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_boxes(boxes, transform):
    yaw = torch.atan2(transform[1, 0], transform[0, 0])
    return torch.cat((transform_points(boxes[:, :3], transform),
                      boxes[:, 3:6], (boxes[:, 6] + yaw)[:, None]), 1)


def move_keypoints(points, old_boxes, new_boxes):
    """Rigid object motion, preserving local geometry and differentiability."""
    local = points - old_boxes[:, None, :3]
    yaw = new_boxes[:, 6] - old_boxes[:, 6]
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    rotated = torch.stack((c * local[..., 0] - s * local[..., 1],
                           s * local[..., 0] + c * local[..., 1], local[..., 2]), -1)
    return rotated + new_boxes[:, None, :3]


def associate(predicted, observed, radius=5.):
    if not len(predicted) or not len(observed):
        empty = torch.empty(0, dtype=torch.long, device=predicted.device)
        return empty, empty
    distance = torch.cdist(predicted[:, :2], observed[:, :2])
    size = (predicted[:, None, 3:6].clamp_min(.1).log() -
            observed[None, :, 3:6].clamp_min(.1).log()).abs().mean(-1)
    cost = distance + size
    feasible = (distance < radius) & (size < .8)
    rows, cols = linear_sum_assignment(cost.detach().masked_fill(~feasible, 1e6).cpu().numpy())
    rows = torch.as_tensor(rows, device=predicted.device)
    cols = torch.as_tensor(cols, device=predicted.device)
    keep = feasible[rows, cols]
    return rows[keep], cols[keep]


def fit_se2(source, target, weights):
    weights = weights / weights.sum().clamp_min(1e-8)
    a = (source[:, :2] * weights[:, None]).sum(0)
    b = (target[:, :2] * weights[:, None]).sum(0)
    x, y = source[:, :2] - a, target[:, :2] - b
    u, singular, vh = torch.linalg.svd(x.T @ (weights[:, None] * y))
    sign = torch.eye(2, device=x.device, dtype=x.dtype)
    sign[1, 1] = torch.det(vh.T @ u.T)
    rot = vh.T @ sign @ u.T
    result = torch.eye(4, device=x.device, dtype=x.dtype)
    result[:2, :2] = rot
    result[:2, 3] = b - rot @ a
    return result, singular


@torch.no_grad()
def register_background(source, target, radius=2., max_yaw=.15):
    """Mutual local matches, robust planar fit, reject unsupported corrections."""
    xyz = source['xyz']
    identity = torch.eye(4, device=xyz.device, dtype=xyz.dtype)
    empty = {'accepted': 0., 'matches': 0., 'residual': 0., 'confidence': 0.}
    if len(xyz) < 6 or len(target['xyz']) < 6:
        return identity, empty
    distance = torch.cdist(xyz[:, :2], target['xyz'][:, :2])
    descriptor = 1 - source['descriptor'] @ target['descriptor'].T
    cost = distance + .25 * descriptor
    cost = cost.masked_fill(distance > radius, 1e6)
    nearest = cost.argmin(1)
    back = cost.argmin(0)
    ids = torch.arange(len(xyz), device=xyz.device)
    keep = (back[nearest] == ids) & (distance[ids, nearest] < radius)
    if int(keep.sum()) < 6:
        return identity, empty
    a, b = xyz[keep], target['xyz'][nearest[keep]]
    weights = source['quality'][keep] * target['quality'][nearest[keep]]
    transform, singular = fit_se2(a, b, weights)
    errors = (transform_points(a, transform)[:, :2] - b[:, :2]).norm(dim=1)
    weights = weights / (1 + (errors / .3).square())
    transform, singular = fit_se2(a, b, weights)
    error = (transform_points(a, transform)[:, :2] - b[:, :2]).norm(dim=1).mean()
    before = (a[:, :2] - b[:, :2]).norm(dim=1).mean()
    angle = torch.atan2(transform[1, 0], transform[0, 0]).abs()
    accepted = bool(singular[-1] > .02 and angle < max_yaw and error < .5 and
                    keep.float().mean() >= .05 and
                    transform[:2, 3].norm() < radius and error <= before + 1e-5)
    confidence = float(keep.float().mean() * torch.exp(-error)) if accepted else 0.
    return (transform if accepted else identity), {
        'accepted': float(accepted), 'matches': float(keep.sum()),
        'residual': float(error), 'confidence': confidence}

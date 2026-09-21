"""SFR geometry: hwl boxes, metric xy, forward rigid transport, inverse map warp."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def se2(delta):
    x, y, yaw = delta.unbind(-1)
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, x, s, c, z, y, z, z, o, z, z, z, z, o), -1).reshape(*delta.shape[:-1], 4, 4)


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_boxes(boxes, transform):
    yaw = boxes[:, 6] + torch.atan2(transform[1, 0], transform[0, 0])
    return torch.cat((transform_points(boxes[:, :3], transform), boxes[:, 3:6], yaw[:, None]), 1)


def transport(points, old_boxes, new_boxes):
    """One rigid transform per point's instance, never resample cached features."""
    local = points - old_boxes[:, :3]
    angle = new_boxes[:, 6] - old_boxes[:, 6]
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*local[:, 0]-s*local[:, 1], s*local[:, 0]+c*local[:, 1], local[:, 2]), -1) + new_boxes[:, :3]


def cell_centers(height, width, bounds, like):
    x = (torch.arange(width, device=like.device, dtype=like.dtype)+.5)*(bounds[3]-bounds[0])/width+bounds[0]
    y = (torch.arange(height, device=like.device, dtype=like.dtype)+.5)*(bounds[4]-bounds[1])/height+bounds[1]
    yy, xx = torch.meshgrid(y, x)
    return torch.stack((xx, yy, torch.zeros_like(xx)), -1)


def warp_map(source, source_to_target, bounds, shape=None):
    """NCHW map in source coordinates -> target grid; align_corners=False."""
    h, w = shape or source.shape[-2:]
    xyz = cell_centers(h, w, bounds, source).reshape(-1, 3)
    src = transform_points(xyz, torch.inverse(source_to_target))
    grid = torch.stack((2*(src[:, 0]-bounds[0])/(bounds[3]-bounds[0])-1,
                        2*(src[:, 1]-bounds[1])/(bounds[4]-bounds[1])-1), -1)
    return F.grid_sample(source, grid.reshape(1, h, w, 2), align_corners=False)


def splat(points, features, bounds, shape, weights=None):
    """Bilinear scatter with normalized conflicts; indices discrete, weights differentiable."""
    h, w = shape
    weights = features.new_ones(len(points)) if weights is None else weights
    xy = torch.stack(((points[:, 0]-bounds[0])*w/(bounds[3]-bounds[0])-.5,
                      (points[:, 1]-bounds[1])*h/(bounds[4]-bounds[1])-.5), -1)
    base = xy.floor()
    fraction = xy-base
    values = features.new_zeros(h*w, features.shape[-1])
    mass = features.new_zeros(h*w, 1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        index = base.long() + base.new_tensor([dx, dy], dtype=torch.long)
        valid = (index[:, 0] >= 0) & (index[:, 0] < w) & (index[:, 1] >= 0) & (index[:, 1] < h)
        blend = (fraction[:, 0] if dx else 1-fraction[:, 0]) * (fraction[:, 1] if dy else 1-fraction[:, 1]) * weights
        flat = index[valid, 1]*w+index[valid, 0]
        values = values.index_add(0, flat, features[valid]*blend[valid, None])
        mass = mass.index_add(0, flat, blend[valid, None])
    result = values/mass.clamp_min(1e-6)
    return result.T.reshape(1, -1, h, w), mass.T.reshape(1, 1, h, w)


@torch.no_grad()
def associate(predicted, observed, radius=5.):
    if not len(predicted) or not len(observed):
        empty = torch.empty(0, dtype=torch.long, device=predicted.device)
        return empty, empty
    distance = torch.cdist(predicted[:, :2], observed[:, :2])
    size = (predicted[:, None, 3:6].clamp_min(.1).log()-observed[None, :, 3:6].clamp_min(.1).log()).abs().mean(-1)
    # Heading is axial for association: pi flips need not break an object track.
    angle = 1-torch.cos(2*(predicted[:, None, 6]-observed[None, :, 6]))
    feasible = (distance <= radius) & (size < .8)
    cost = (distance+size+.2*angle).masked_fill(~feasible, 1e6)
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    rows, cols = torch.as_tensor(rows, device=predicted.device), torch.as_tensor(cols, device=predicted.device)
    keep = feasible[rows, cols]
    return rows[keep], cols[keep]

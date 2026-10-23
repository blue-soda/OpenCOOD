"""Same nearest SE(2) warp, with reusable geometry and grid-side invalidation.

Only static cell coordinates persist. Feature states, poses and sampling grids
remain frame-owned. Mask invalid sampling coordinates before grid_sample rather
than multiplying every output feature channel by the coverage mask afterwards.
"""
import numpy as np
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import planar_pose


class CachedMetricWarp:
    def __init__(self, bounds):
        self.bounds = tuple(bounds)
        self.geometry = {}

    @property
    def geometry_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.geometry.values())

    def __call__(self, value, past_pose, current_pose, bounds):
        if tuple(bounds) != self.bounds or value.ndim != 4:
            raise ValueError('Expected fixed metric bounds and NCHW features')
        b, _, h, w = value.shape
        if min(b, h, w) < 1 or not value.is_floating_point():
            raise ValueError('Nonempty floating features required')
        key = (h, w, value.device, value.dtype)
        xmin, ymin, _, xmax, ymax, _ = self.bounds
        if key not in self.geometry:
            # Bound incidental changes of device/shape; normal detector has one
            # spatial/deep grid. Never retain arbitrary per-frame geometries.
            if len(self.geometry) >= 4:
                self.geometry.clear()
            xs = xmin + (torch.arange(w, device=value.device, dtype=value.dtype)+.5)*((xmax-xmin)/w)
            ys = ymin + (torch.arange(h, device=value.device, dtype=value.dtype)+.5)*((ymax-ymin)/h)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self.geometry[key] = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        matrix = np.linalg.solve(planar_pose(past_pose), planar_pose(current_pose))
        matrix = torch.as_tensor(matrix, device=value.device, dtype=value.dtype)
        source = torch.matmul(self.geometry[key], matrix.t())
        grid = torch.stack([2*(source[...,0]-xmin)/(xmax-xmin)-1,
                            2*(source[...,1]-ymin)/(ymax-ymin)-1], dim=-1)
        grid = grid.unsqueeze(0).expand(b,-1,-1,-1)
        valid = (grid.abs() <= 1).all(dim=-1).unsqueeze(1)
        # Coordinates 2,2 are strictly outside even a 1x1 grid with nearest
        # sampling. This preserves original valid semantics at [-1,1] edges.
        grid = grid.masked_fill(~valid.permute(0,2,3,1), 2.)
        result = F.grid_sample(value, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        return result, valid

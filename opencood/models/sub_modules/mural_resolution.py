"""DAIR port of MURAL's shared convolution / resolution-aware BN core.

Reference: Soyyigit et al., RTCSA 2025, section IV-B, equation (4).
This module is not the author's TensorRT scheduler or full runtime.
"""
import copy
import torch
from torch import nn


class ResolutionBatchNorm(nn.Module):
    def __init__(self, source, count):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(source) for _ in range(count)])
        self.index = 0

    def forward(self, value):
        return self.layers[self.index](value)


def install_resolution_bn(module, count):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, ResolutionBatchNorm(child, count))
        else:
            install_resolution_bn(child, count)


def select_resolution(model, config, index):
    for child in model.modules():
        if isinstance(child, ResolutionBatchNorm):
            child.index = index
    args = config['model']['args']
    size, bounds = args['voxel_size'], args['lidar_range']
    vfe = model.pillar_vfe
    vfe.voxel_x, vfe.voxel_y, vfe.voxel_z = size
    vfe.x_offset, vfe.y_offset, vfe.z_offset = [s / 2 + b for s, b in zip(size, bounds)]
    grid = args['point_pillar_scatter']['grid_size']
    model.scatter.nx, model.scatter.ny, model.scatter.nz = map(int, grid)
    model.scatter.model_cfg = args['point_pillar_scatter']


def resolution_configs(base, grids=((504, 200), (448, 176), (400, 160))):
    result = []
    for nx, ny in grids:
        if nx % 8 or ny % 8:
            raise ValueError('Grid must match all three backbone strides')
        c = copy.deepcopy(base)
        bounds = c['preprocess']['cav_lidar_range']
        size = [(bounds[3]-bounds[0])/nx, (bounds[4]-bounds[1])/ny, bounds[5]-bounds[2]]
        c['preprocess']['args']['voxel_size'] = size
        c['model']['args']['voxel_size'] = size
        c['model']['args']['lidar_range'] = bounds
        c['model']['args']['point_pillar_scatter']['grid_size'] = [nx, ny, 1]
        c['postprocess']['anchor_args'].update(W=nx, H=ny, D=1, vw=size[0], vh=size[1], vd=size[2])
        result.append(c)
    return result


class OperationCounter:
    """Executed Conv2d/ConvTranspose2d/Linear MACs; excludes other operations."""
    def __init__(self, model):
        self.macs = 0
        self.handles = []
        def hook(m, inputs, output):
            if isinstance(m, nn.Conv2d):
                self.macs += output.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.ConvTranspose2d):
                self.macs += inputs[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            else:
                self.macs += output.numel() * m.in_features
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                self.handles.append(m.register_forward_hook(hook))

    def reset(self):
        self.macs = 0

    def close(self):
        for handle in self.handles:
            handle.remove()

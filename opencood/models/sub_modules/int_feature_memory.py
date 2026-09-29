"""INT feature memory port (Concat / GRU) with metric BEV alignment.

Fusion equations and layer construction adapted from ADLab-AutoDrive/INT,
commit 988157ff131a0c027472bd0f00c0bda0e08cded0, voxelnet.FusionModule.
Copyright (c) 2021-2022 Alibaba Group. Apache-2.0; see third_party/INT.
Changes: explicit real delta_t, separate state ownership, and geometric warp.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class INTLayer(nn.Sequential):
    def add(self, layer):
        self.add_module(str(len(self)), layer)


def conv_spec(inputs, outputs, activation=None, num_blocks=0):
    return dict(inplanes=inputs, planes=outputs, num_blocks=num_blocks,
                kernel_size=3, activation=activation)


def make_fusion_config(mode, channels=64):
    cfg = dict(type='concat' if mode == 'concat' else 'infinite_GRU',
               sub_type=None, input_channels=channels, nsweeps=1,
               spatial_weight_conv=None, cur_pre_fusion_conv=None,
               past_pre_fusion_conv=None)
    if mode == 'concat':
        # Full-width branches permit identity initialization from single-frame P1.
        cfg.update(cur_pre_fusion_conv=conv_spec(channels, channels),
                   past_pre_fusion_conv=conv_spec(channels, channels),
                   post_fusion_conv=conv_spec(2 * channels, channels))
    elif mode == 'gru':
        cfg.update(reset_gate_conv=conv_spec(2 * channels, channels, nn.Sigmoid()),
                   update_gate_conv=conv_spec(2 * channels, channels, nn.Sigmoid()),
                   candidate_conv=conv_spec(2 * channels, channels, nn.Tanh()),
                   post_fusion_conv=conv_spec(channels + 2, channels))
    else:
        raise ValueError(mode)
    return cfg


class INTFusionModule(nn.Module):
    """Same parameter names and tensor operations as the supported INT paths."""
    def __init__(self, fusion_method):
        super().__init__()
        self.fusion_method = fusion_method
        for name in ['cur_pre_fusion_conv', 'past_pre_fusion_conv', 'post_fusion_conv']:
            spec = fusion_method[name]
            setattr(self, name, self._make_layer(**spec) if spec else None)
        if fusion_method['type'] == 'infinite_GRU':
            for name in ['reset_gate_conv', 'update_gate_conv', 'candidate_conv']:
                setattr(self, name, self._make_layer(**fusion_method[name]))

    @staticmethod
    def _make_layer(inplanes, planes, num_blocks, stride=1, kernel_size=3, activation=None):
        activation = nn.ReLU() if activation is None else activation
        padding = 0 if kernel_size == 1 else 1
        layers = INTLayer(nn.ZeroPad2d(padding),
                          nn.Conv2d(inplanes, planes, kernel_size, stride=stride, bias=False),
                          nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01), activation)
        for _ in range(num_blocks):
            layers.add(nn.Conv2d(planes, planes, kernel_size, padding=padding, bias=False))
            layers.add(nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01))
            layers.add(activation)
        return layers

    def forward(self, x, past_x_list, delta_t=0.05):
        if self.cur_pre_fusion_conv is not None:
            x = self.cur_pre_fusion_conv(x)
        if self.past_pre_fusion_conv is not None:
            past_x_list = [self.past_pre_fusion_conv(p) for p in past_x_list]
        cached = None
        if self.fusion_method['type'] == 'concat':
            out = torch.cat([x] + past_x_list, dim=1)
        else:
            assert len(past_x_list) == 1
            # Clone age: callers own their input state, including diagnostic reuse.
            past_feat = past_x_list[0][:, :-1]
            past_time = past_x_list[0][:, -1].clone()
            past_time[past_feat.sum(dim=1) != 0] += float(delta_t)
            reset = self.reset_gate_conv(torch.cat([x, past_feat], dim=1))
            update = self.update_gate_conv(torch.cat([x, past_feat], dim=1))
            candidate = self.candidate_conv(torch.cat([x, reset * past_feat], dim=1))
            hidden = update * past_feat + (1 - update) * candidate
            out = torch.cat([hidden, past_time.unsqueeze(1)], dim=1)
            cached = out.detach()
            with torch.no_grad():
                mask = (x.sum(dim=1) != 0).to(x.dtype).unsqueeze(1)
            out = torch.cat([out, mask], dim=1)
        if self.post_fusion_conv is not None:
            out = self.post_fusion_conv(out)
        return out, cached

    def initialize_concat_identity(self):
        if self.fusion_method['type'] != 'concat':
            return
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.dirac_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, math.sqrt(1 + layer.eps))
                nn.init.zeros_(layer.bias)
                layer.running_mean.zero_()
                layer.running_var.fill_(1)


def planar_pose(pose):
    """SE(2) projection of dataset LiDAR-to-world pose, not full SE(3)."""
    pose = np.asarray(pose, dtype=np.float64)
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, pose[0, 3]], [s, c, pose[1, 3]], [0, 0, 1]], dtype=np.float64)


def warp_bev(value, past_pose, current_pose, lidar_range, mode='nearest'):
    """Inverse sampling at metric cell centers, valid for non-square BEV ranges."""
    b, _, h, w = value.shape
    xmin, ymin, _, xmax, ymax, _ = lidar_range
    xs = xmin + (torch.arange(w, device=value.device, dtype=value.dtype) + .5) * ((xmax-xmin)/w)
    ys = ymin + (torch.arange(h, device=value.device, dtype=value.dtype) + .5) * ((ymax-ymin)/h)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    xy1 = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    matrix = np.linalg.solve(planar_pose(past_pose), planar_pose(current_pose))
    matrix = torch.as_tensor(matrix, device=value.device, dtype=value.dtype)
    source = torch.matmul(xy1, matrix.t())
    grid = torch.stack([2*(source[..., 0]-xmin)/(xmax-xmin)-1,
                        2*(source[..., 1]-ymin)/(ymax-ymin)-1], dim=-1)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    valid = (grid.abs() <= 1).all(dim=-1).unsqueeze(1)
    warped = F.grid_sample(value, grid, mode=mode, padding_mode='zeros', align_corners=False)
    return warped * valid.to(value.dtype), valid


class INTFeatureMemory(nn.Module):
    """Explicit scene-owned state. batch=1 is deliberate for this stream runner."""
    def __init__(self, mode, channels, lidar_range, max_gap_s=.25, identity_init=True):
        super().__init__()
        self.mode, self.channels = mode, channels
        self.lidar_range, self.max_gap_s = lidar_range, max_gap_s
        self.fusion = INTFusionModule(make_fusion_config(mode, channels)) if mode != 'single' else None
        if self.fusion is not None and identity_init:
            self.fusion.initialize_concat_identity()

    def step(self, current, state, meta, reset=False, align=True):
        assert current.shape[0] == 1, 'Use one explicit state per sequence, not batch slots.'
        reason = 'initial' if state is None else None
        delta_t = 0.
        if state is not None and not reset:
            if state['scene'] != meta['segment']:
                reason = 'scene'
            else:
                delta_t = (int(meta['timestamp_us']) - state['timestamp_us']) / 1e6
                if delta_t <= 0:
                    raise ValueError('Non-causal or repeated frame in one stream')
                if delta_t > self.max_gap_s:
                    reason = 'gap'
        if reset:
            reason = 'explicit'
        if self.mode == 'single':
            return current, None, {'reset': 'single', 'delta_t': delta_t, 'state_bytes': 0, 'warp': False}
        if reason is not None:
            width = self.channels + int(self.mode == 'gru')
            past = current.new_zeros((1, width, *current.shape[2:]))
            coverage = 0.
        elif align:
            past, valid = warp_bev(state['value'], state['pose'], meta['pose'], self.lidar_range)
            coverage = float(valid.float().mean())
        else:
            past = state['value'].clone()
            coverage = 1.
        output, cached = self.fusion(current, [past], delta_t=delta_t)
        value = output.detach() if cached is None else cached
        new_state = dict(value=value, scene=meta['segment'], pose=meta['pose'],
                         timestamp_us=int(meta['timestamp_us']))
        info = dict(reset=reason, delta_t=delta_t, coverage=coverage,
                    state_bytes=value.numel()*value.element_size(),
                    warp=reason is None and align)
        return output, new_state, info

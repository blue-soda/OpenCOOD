"""Migrated shallow encoder and radius-four high-level state update.

Stages A/B use full updates. Region selection is deliberately not implemented in
this training baseline. No teacher, original GRU, decoder or shrink is retained.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class MigratedStateUpdate(nn.Module):
    radius = 4
    evidence_channels = 128
    state_channels = 256

    def __init__(self, source_stage):
        super().__init__()
        if len(source_stage) < 2:
            raise ValueError('Migration needs two source residual blocks')
        self.input_projection = nn.Conv2d(385, 128, 1)
        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.input_projection.bias.zero_()
            for channel in range(128):
                self.input_projection.weight[channel, channel, 0, 0] = 1
        self.blocks = copy.deepcopy(nn.Sequential(*list(source_stage)[:2]))
        first, second = self.blocks
        if (first.conv1.in_channels != 128 or first.conv2.out_channels != 256 or
                first.conv1.stride != (2, 2) or first.downsample is None or
                second.conv1.stride != (1, 1) or second.conv2.out_channels != 256):
            raise ValueError('Unexpected source spatial architecture')
        for block in self.blocks:
            for conv in (block.conv1, block.conv2):
                if conv.kernel_size != (3, 3) or conv.padding != (1, 1) or conv.dilation != (1, 1) or conv.groups != 1:
                    raise ValueError('Unsupported local dependency')
        first.conv1.stride = (1, 1)
        first.downsample[0].stride = (1, 1)
        first.stride = 1

    def forward(self, packed):
        return self.blocks(self.input_projection(packed))


class LocalStateDetector(nn.Module):
    version = 'migrated_local_state_v1'

    def __init__(self, teacher, config, memory_policy='reset'):
        super().__init__()
        if memory_policy not in ('reset', 'recurrent'):
            raise ValueError('Unknown memory policy')
        if teacher.cls_head.in_channels != 256 or teacher.backbone.use_dropout:
            raise ValueError('Expected a deterministic 256-channel teacher readout')
        self.memory_policy = memory_policy
        self.bounds = list(config['model']['args']['lidar_range'])
        self.pillar_vfe = copy.deepcopy(teacher.pillar_vfe)
        self.scatter = copy.deepcopy(teacher.scatter)
        self.observation = copy.deepcopy(teacher.backbone.resnet.layer0)
        if self.observation[-1].conv2.out_channels != 128:
            raise ValueError('Expected 128-channel shallow encoder')
        self.update = MigratedStateUpdate(teacher.backbone.resnet.layer1)
        self.cls_head = copy.deepcopy(teacher.cls_head)
        self.reg_head = copy.deepcopy(teacher.reg_head)
        self.dir_head = copy.deepcopy(teacher.dir_head) if teacher.use_dir else None
        # The teacher is frozen by the runner; copied spatial weights must learn.
        for p in self.parameters():
            p.requires_grad_(True)
        for p in self.pillar_vfe.parameters():
            p.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.pillar_vfe.eval()
        self.scatter.eval()
        # Fixed statistics make crop execution independent of the image extent.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_head_trainable(self, enabled):
        for module in (self.cls_head, self.reg_head, self.dir_head):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(enabled)

    def spatial(self, data):
        p = data['processed_lidar']
        with torch.no_grad():
            if p['voxel_features'].shape[0]:
                batch = {k: p[k] for k in ['voxel_features', 'voxel_coords', 'voxel_num_points']}
                batch['batch_size'] = 1
                return self.scatter(self.pillar_vfe(batch))['spatial_features']
            nx, ny, _ = self.scatter.model_cfg['grid_size']
            return self.cls_head.weight.new_zeros((1, 64, int(ny), int(nx)))

    def step_spatial(self, spatial, state, meta, detach_state=True):
        if spatial.shape[0] != 1:
            raise ValueError('One explicit sequence state required')
        evidence = self.observation(spatial)
        lattice = block_lidar_range(self.bounds, spatial.shape[-2:], evidence.shape[-2:], 2)
        reason = 'initial' if state is None else None
        if self.memory_policy == 'reset':
            reason = 'single_frame_stage_a'
        elif meta.get('reset_before', False):
            reason = 'explicit'
        elif state is not None:
            if state['scene'] != meta['segment']:
                reason = 'scene'
            else:
                gap = (int(meta['timestamp_us']) - state['timestamp_us']) / 1e6
                if gap <= 0:
                    raise ValueError('Noncausal local-state stream')
                if gap > .25:
                    reason = 'gap'
        if reason is None:
            if state['grid_range'] != lattice or state['value'].shape[-2:] != evidence.shape[-2:]:
                raise ValueError('Changed state lattice inside a sequence')
            old, valid = warp_bev(state['value'], state['pose'], meta['pose'], lattice)
        else:
            old = evidence.new_zeros((1, 256, *evidence.shape[-2:]))
            valid = torch.zeros_like(old[:, :1], dtype=torch.bool)
        value = self.update(torch.cat([evidence, old, valid.to(old.dtype)], dim=1))
        pred = dict(psm=self.cls_head(value), rm=self.reg_head(value))
        if self.dir_head is not None:
            pred['dm'] = self.dir_head(value)
        new = dict(value=value.detach() if detach_state else value, pose=meta['pose'],
                   scene=meta['segment'], timestamp_us=int(meta['timestamp_us']), grid_range=lattice)
        size = value.numel() * value.element_size()
        info = dict(reset=reason, cache_bytes=size, state_bytes=size,
                    memory_policy=self.memory_policy, dense_update=True,
                    cache_stages=[dict(actual_write_fraction=1., executed_input_area_ratio=1.) for _ in range(3)])
        return pred, new, info

    def step(self, data, state, meta, detach_state=True):
        return self.step_spatial(self.spatial(data), state, meta, detach_state)


def feature_distillation(student, teacher, teacher_predictions):
    """Frozen channel-scale normalization; equal global and confidence-weighted terms."""
    if student.shape != teacher.shape:
        raise ValueError('Teacher/student detection features must share shape')
    target = teacher.detach()
    scale = target.square().mean((0, 2, 3), keepdim=True).sqrt().clamp_min(1.)
    error = F.smooth_l1_loss(student / scale, target / scale, reduction='none').mean(1, keepdim=True)
    confidence = teacher_predictions['psm'].detach().sigmoid().amax(1, keepdim=True).square()
    return .5 * error.mean() + .5 * (error * confidence).sum() / confidence.sum().clamp_min(1.)

"""Accuracy-first recurrent feature pyramid with the complete mature spatial path."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_two_scale import TwoScaleDetector, StateUpdate
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class PyramidCore(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.observation = copy.deepcopy(source.backbone.resnet.layer0)
        self.coarse_encoder = copy.deepcopy(source.backbone.resnet.layer1)
        self.deep_encoder = copy.deepcopy(source.backbone.resnet.layer2)
        self.deblocks = copy.deepcopy(source.backbone.deblocks)
        self.readout = copy.deepcopy(source.shrink_conv) if source.shrink_flag else nn.Identity()
        self.cls_head = copy.deepcopy(source.cls_head)
        self.reg_head = copy.deepcopy(source.reg_head)
        self.dir_head = copy.deepcopy(source.dir_head) if source.use_dir else None
        self.fine_update = StateUpdate(128)
        self.coarse_update = StateUpdate(256)
        self.deep_update = StateUpdate(512)


class PyramidStateDetector(TwoScaleDetector):
    version = 'pyramid_state_accuracy_v1'
    state_keys = ('value', 'coarse', 'deep')

    def __init__(self, teacher, config, memory_policy='recurrent'):
        nn.Module.__init__(self)
        if memory_policy not in ('reset', 'recurrent'):
            raise ValueError(memory_policy)
        if len(teacher.backbone.deblocks) != 3:
            raise ValueError('This contract requires three mature pyramid readout branches')
        self.variant = self.version
        self.memory_policy = memory_policy
        self.spatial_only = False  # Engineering intervention, never the training policy.
        self.bounds = config['model']['args']['lidar_range']
        self.pillar_vfe = copy.deepcopy(teacher.pillar_vfe)
        self.scatter = copy.deepcopy(teacher.scatter)
        self.core = PyramidCore(teacher)
        for p in self.parameters():
            p.requires_grad_(True)
        for p in self.pillar_vfe.parameters():
            p.requires_grad_(False)
        self.train(False)

    def updates_are_initial_identity(self):
        return all(torch.count_nonzero(p).item() == 0
                   for update in (self.core.fine_update, self.core.coarse_update, self.core.deep_update)
                   for p in (update.history.weight, update.residual[-1].weight, update.residual[-1].bias))

    def migrated_source_pairs(self, teacher):
        pairs = [(self.core.observation, teacher.backbone.resnet.layer0),
                 (self.core.coarse_encoder, teacher.backbone.resnet.layer1),
                 (self.core.deep_encoder, teacher.backbone.resnet.layer2),
                 (self.core.deblocks, teacher.backbone.deblocks),
                 (self.core.cls_head, teacher.cls_head), (self.core.reg_head, teacher.reg_head)]
        if teacher.shrink_flag:
            pairs.append((self.core.readout, teacher.shrink_conv))
        if teacher.use_dir:
            pairs.append((self.core.dir_head, teacher.dir_head))
        return pairs

    def step_spatial(self, spatial, state, meta, detach_state=True):
        if spatial.shape[0] != 1:
            raise ValueError('One stream required')
        reason = 'initial' if state is None else None
        if self.memory_policy == 'reset' or meta.get('reset_before', False):
            reason = 'explicit'
        if state is not None and reason is None:
            if state['scene'] != meta['segment']:
                reason = 'scene'
            else:
                gap = (int(meta['timestamp_us']) - state['timestamp_us']) / 1e6
                if gap <= 0:
                    raise ValueError('Noncausal state')
                if gap > .25:
                    reason = 'gap'
        x = spatial
        values = []
        for key, encoder, update, stride in (
                ('value', self.core.observation, self.core.fine_update, 2),
                ('coarse', self.core.coarse_encoder, self.core.coarse_update, 4),
                ('deep', self.core.deep_encoder, self.core.deep_update, 8)):
            current = encoder(x)
            if self.spatial_only:
                x = current
            else:
                if reason:
                    old = torch.zeros_like(current)
                    valid = torch.zeros_like(current[:, :1], dtype=torch.bool)
                else:
                    bounds = block_lidar_range(self.bounds, spatial.shape[-2:], current.shape[-2:], stride)
                    old, valid = warp_bev(state[key], state['pose'], meta['pose'], bounds)
                x = update(current, old, valid)
            values.append(x)
        ups = [deblock(value) for deblock, value in zip(self.core.deblocks, values)]
        if len({tuple(x.shape[-2:]) for x in ups}) != 1:
            raise ValueError('Input lattice does not align the mature pyramid upsampling branches')
        features = self.core.readout(torch.cat(ups, dim=1))
        prediction = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None:
            prediction['dm'] = self.core.dir_head(features)
        new = {key: value.detach() if detach_state else value
               for key, value in zip(self.state_keys, values)}
        new.update(pose=meta['pose'], scene=meta['segment'], timestamp_us=int(meta['timestamp_us']))
        size = sum(x.numel() * x.element_size() for x in values)
        return prediction, new, dict(reset=reason, state_bytes=size, cache_bytes=size,
            cache_stages=[dict(actual_write_fraction=1., executed_input_area_ratio=1.) for _ in values])

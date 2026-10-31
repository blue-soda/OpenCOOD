"""Dense two-scale state baseline. Selective execution is a later experiment."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class StateUpdate(nn.Module):
    radius = 2

    def __init__(self, channels):
        super().__init__()
        self.history = nn.Conv2d(channels + 1, channels, 1, bias=False)
        nn.init.zeros_(self.history.weight)
        width = channels // 2
        self.residual = nn.Sequential(nn.Conv2d(channels, width, 1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, channels, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, current, old, valid):
        x = current + self.history(torch.cat((old, valid.to(old.dtype)), 1))
        return F.relu(x + self.residual(x))


class BottleneckReadout(nn.Module):
    """Pointwise fusion with a zero-initialized spatial residual."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(256, 256, 1)
        self.residual = nn.Sequential(nn.Conv2d(256, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 256, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        x = F.relu(self.projection(x))
        return F.relu(x + self.residual(x))


class ContextResidual(nn.Module):
    """Spatial capacity at stride eight, without adding recurrent state."""
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(384, 192, 1), nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(), nn.Conv2d(192, 384, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return F.relu(x + self.residual(x))


class CoarseContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(256, 384, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ContextResidual() for _ in range(3)])
        self.projection = nn.Conv2d(384, 256, 1)
        # Preserve the small model's initial function, including its coarse state.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        context = self.projection(self.blocks(F.relu(self.down(x))))
        return x + F.interpolate(context, size=x.shape[-2:], mode='nearest')


class TwoScaleCore(nn.Module):
    def __init__(self, teacher, architecture='two_scale_v1'):
        super().__init__()
        source = teacher.backbone.resnet.layer0
        self.observation = copy.deepcopy(source if architecture in ('two_scale_full', 'two_scale_medium') else source[0])
        self.coarse_encoder = copy.deepcopy(nn.Sequential(*list(teacher.backbone.resnet.layer1)[:2]))
        self.fine_update = StateUpdate(128)
        self.coarse_update = StateUpdate(256)
        self.coarse_projection = nn.Conv2d(256, 128, 1)
        self.readout = (nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())
                        if architecture == 'two_scale_v1' else BottleneckReadout())
        self.cls_head = copy.deepcopy(teacher.cls_head)
        self.reg_head = copy.deepcopy(teacher.reg_head)
        self.dir_head = copy.deepcopy(teacher.dir_head) if teacher.use_dir else None
        # Construct after all shared modules so seed303 preserves their weights.
        self.coarse_context = CoarseContext() if architecture == 'two_scale_medium' else nn.Identity()


class TwoScaleDetector(nn.Module):
    version = 'two_scale_dense_v1'

    def __init__(self, teacher, config, memory_policy='recurrent', architecture='two_scale_v1'):
        super().__init__()
        if memory_policy not in ('reset', 'recurrent'): raise ValueError(memory_policy)
        versions = dict(two_scale_v1='two_scale_dense_v1',
                        two_scale_narrow='two_scale_narrow_reader_v1',
                        two_scale_full='two_scale_full_encoder_v1',
                        two_scale_medium='two_scale_medium_context_v1')
        if architecture not in versions: raise ValueError(architecture)
        self.version = versions[architecture]
        self.memory_policy = memory_policy
        self.variant = self.version
        self.bounds = config['model']['args']['lidar_range']
        self.pillar_vfe = copy.deepcopy(teacher.pillar_vfe)
        self.scatter = copy.deepcopy(teacher.scatter)
        self.core = TwoScaleCore(teacher, architecture)
        for p in self.parameters(): p.requires_grad_(True)
        for p in self.pillar_vfe.parameters(): p.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.pillar_vfe.eval()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        return self

    def step_spatial(self, spatial, state, meta, detach_state=True):
        if spatial.shape[0] != 1: raise ValueError('One stream required')
        reason = 'initial' if state is None else None
        if self.memory_policy == 'reset' or meta.get('reset_before', False): reason = 'explicit'
        if state is not None and reason is None:
            if state['scene'] != meta['segment']: reason = 'scene'
            else:
                gap = (int(meta['timestamp_us']) - state['timestamp_us']) / 1e6
                if gap <= 0: raise ValueError('Noncausal state')
                if gap > .25: reason = 'gap'
        fine = self.core.observation(spatial)
        coarse = self.core.coarse_context(self.core.coarse_encoder(fine))
        values = []
        for name, current, update, stride in (
                ('value', fine, self.core.fine_update, 2),
                ('coarse', coarse, self.core.coarse_update, 4)):
            if reason:
                old = torch.zeros_like(current)
                valid = torch.zeros_like(current[:, :1], dtype=torch.bool)
            else:
                bounds = block_lidar_range(self.bounds, spatial.shape[-2:], current.shape[-2:], stride)
                old, valid = warp_bev(state[name], state['pose'], meta['pose'], bounds)
            values.append(update(current, old, valid))
        fine, coarse = values
        context = F.interpolate(self.core.coarse_projection(coarse), size=fine.shape[-2:], mode='nearest')
        features = self.core.readout(torch.cat((fine, context), 1))
        prediction = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None: prediction['dm'] = self.core.dir_head(features)
        new = dict(value=fine.detach() if detach_state else fine,
                   coarse=coarse.detach() if detach_state else coarse,
                   pose=meta['pose'], scene=meta['segment'], timestamp_us=int(meta['timestamp_us']))
        size = sum(x.numel()*x.element_size() for x in values)
        return prediction, new, dict(reset=reason, state_bytes=size, cache_bytes=size,
            cache_stages=[dict(actual_write_fraction=1., executed_input_area_ratio=1.) for _ in range(3)])

    def step(self, data, state, meta, detach_state=True):
        p = data['processed_lidar']
        with torch.no_grad():
            if p['voxel_features'].shape[0]:
                batch = {k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')}
                batch['batch_size'] = 1
                spatial = self.scatter(self.pillar_vfe(batch))['spatial_features']
            else:
                nx, ny, _ = self.scatter.model_cfg['grid_size']
                spatial = self.core.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
        return self.step_spatial(spatial,state,meta,detach_state)

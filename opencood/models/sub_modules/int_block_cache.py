"""Pretrained INT bridge: cache each residual block, without new parameters.

Full updates preserve the existing detector. Partial updates recompute selected
block outputs from the mixed preceding layer; unselected outputs are warped
history. This is not equivalent to dense inference, except when all cells update.
Pillar, GRU, multiscale decoder and heads remain dense and must be timed as such.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_feature_memory import warp_bev


def block_contract(backbone):
    """Reject unsupported dependencies rather than silently underestimating halo."""
    if backbone.use_dropout:
        raise ValueError('Dropout is unsupported in the fixed-weight bridge')
    entries = []
    scale = 1
    for stage in range(backbone.num_levels):
        for index, block in enumerate(getattr(backbone.resnet, 'layer%d' % stage)):
            if not isinstance(block, BasicBlock):
                raise ValueError('Only the checkpoint-compatible BasicBlock is supported')
            a, b = block.conv1, block.conv2
            stride = a.stride[0]
            if (stride not in (1, 2) or a.stride != (stride, stride) or
                    b.stride != (1, 1) or any(c.kernel_size != (3, 3) or
                    c.padding != (1, 1) or c.dilation != (1, 1) or
                    c.groups != 1 for c in (a, b))):
                raise ValueError('Unsupported spatial block dependency')
            if any(not isinstance(m, nn.BatchNorm2d) or not m.track_running_stats
                   for m in (block.bn1, block.bn2)):
                raise ValueError('Fixed-statistic BatchNorm is required')
            if block.downsample is not None:
                down = block.downsample
                if (len(down) != 2 or not isinstance(down[0], nn.Conv2d) or
                        down[0].kernel_size != (1, 1) or down[0].stride != a.stride or
                        down[0].padding != (0, 0) or down[0].groups != 1 or
                        not isinstance(down[1], nn.BatchNorm2d) or not down[1].track_running_stats):
                    raise ValueError('Unsupported residual projection')
            scale *= stride
            entries.append(dict(stage=stage, block=index, stride=stride,
                                scale=scale, radius=1 + stride))
    return entries


class BlockCachedBackbone:
    """Execution adapter sharing all original parameters; no duplicate module copy."""
    def __init__(self, backbone, tile=8, allow_fallback=True):
        if tile < 1:
            raise ValueError('Positive tile required')
        self.backbone = backbone
        self.contract = block_contract(backbone)
        self.tile, self.allow_fallback = tile, allow_fallback

    def run(self, x, cached=None, requests=None, execution='sparse'):
        if execution not in ('sparse', 'masked_dense'):
            raise ValueError('Unknown execution')
        if any(m.training for m in self.backbone.modules()) or torch.is_grad_enabled() or x.shape[0] != 1:
            raise ValueError('Frozen inference with batch=1 required')
        if (cached is None) != (requests is None):
            raise ValueError('Supply both caches and masks, or neither for full update')
        if cached is not None and (len(cached) != len(self.contract) or len(requests) != len(self.contract)):
            raise ValueError('One cache and mask per residual block required')
        values, writes, stats, stages = [], [], [], []
        for n, entry in enumerate(self.contract):
            block = getattr(self.backbone.resnet, 'layer%d' % entry['stage'])[entry['block']]
            if cached is None:
                x = block(x)
                written = torch.ones_like(x[:, :1], dtype=torch.bool)
                stat = dict(actual_write_fraction=1., executed_input_area_ratio=1.,
                            fallback=False, full_update=True)
            else:
                expected = (1, block.conv2.out_channels,
                            (x.shape[-2] + entry['stride'] - 1) // entry['stride'],
                            (x.shape[-1] + entry['stride'] - 1) // entry['stride'])
                if cached[n].shape != expected or requests[n].shape != (1, 1, *expected[-2:]) or requests[n].dtype != torch.bool:
                    raise ValueError('Cache/mask shape or dtype mismatch')
                x, written, stat = refresh_tiles(block, x, cached[n], requests[n],
                    entry['stride'], entry['radius'], self.tile,
                    allow_fallback=self.allow_fallback, execution=execution)
            values.append(x)
            writes.append(written)
            stats.append(dict(entry, **stat))
            if n + 1 == len(self.contract) or self.contract[n + 1]['stage'] != entry['stage']:
                stages.append(x)
        return stages, values, writes, stats


class BlockCacheINT:
    """Original-weight dense baseline and an explicitly experimental cache mode.

    threshold=None forces full updates. A numeric threshold compares current GRU
    hidden state with the very same aligned hidden state consumed by GRU. No
    additional GRU-history warp or dense spatial output is used by the selector.
    This heuristic has no calibrated operating point yet.
    """
    def __init__(self, model, config, threshold=None, tile=8, max_age=.25,
                 execution='sparse', allow_fallback=True):
        if model.feature_memory.mode != 'gru':
            raise ValueError('The bridge currently supports ANN GRU only')
        if threshold is not None and (threshold < 0 or not float(threshold) < float('inf')):
            raise ValueError('Nonnegative finite threshold required')
        if max_age <= 0 or execution not in ('sparse', 'masked_dense'):
            raise ValueError('Invalid age or execution')
        self.model = model
        self.blocks = BlockCachedBackbone(model.backbone, tile, allow_fallback)
        self.bounds = config['model']['args']['lidar_range']
        self.threshold, self.max_age, self.execution = threshold, max_age, execution

    def step(self, data, state, meta):
        m = self.model
        if m.training or torch.is_grad_enabled():
            raise ValueError('Fixed-weight inference only')
        p = data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch = {k: p[k] for k in ['voxel_features', 'voxel_coords', 'voxel_num_points']}
            batch['batch_size'] = 1
            spatial = m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx, ny, _ = m.scatter.model_cfg['grid_size']
            spatial = m.cls_head.weight.new_zeros((1, 64, int(ny), int(nx)))
        selection = []
        def observe(current, previous, valid):
            energy = torch.maximum(current.square().mean(1, keepdim=True),
                                   previous.square().mean(1, keepdim=True)).clamp_min(1e-6)
            difference = ((current - previous).square().mean(1, keepdim=True) / energy).sqrt()
            selection.append((difference > self.threshold) | ~valid)
        temporal, fm, info = m.feature_memory.step(spatial, None if state is None else state['fm'],
            meta, reset=meta.get('reset_before', False),
            state_observer=observe if self.threshold is not None else None)
        full = state is None or bool(info['reset']) or self.threshold is None
        cached, requests, ages = None, None, []
        if not full:
            if len(selection) != 1:
                raise RuntimeError('Missing aligned GRU observation')
            cached, requests = [], []
            for entry, value, age in zip(self.blocks.contract, state['blocks'], state['ages']):
                packed, valid = warp_bev(torch.cat([value, age], dim=1), state['pose'], meta['pose'], self.bounds)
                old, elapsed = packed[:, :-1], packed[:, -1:] + info['delta_t']
                request = F.max_pool2d(selection[0].float(), entry['scale'], entry['scale'], ceil_mode=True).bool()
                requests.append(request | ~valid | (elapsed >= self.max_age))
                cached.append(old)
                ages.append(elapsed)
        stages, values, writes, stats = self.blocks.run(temporal, cached, requests, self.execution)
        ages = [torch.zeros_like(v[:, :1]) if full else torch.where(w, torch.zeros_like(a), a)
                for v, w, a in zip(values, writes, ages if not full else [None] * len(values))]
        features = m.backbone.decode_multiscale_feature(stages)
        if m.shrink_flag:
            features = m.shrink_conv(features)
        prediction = dict(psm=m.cls_head(features), rm=m.reg_head(features))
        if m.use_dir:
            prediction['dm'] = m.dir_head(features)
        new = dict(fm=fm, blocks=values, ages=ages, pose=meta['pose'])
        stage_stats = []
        for stage in range(m.backbone.num_levels):
            group = [s for s in stats if s['stage'] == stage]
            stage_stats.append({key: sum(s[key] for s in group) / len(group) for key in
                                ['actual_write_fraction', 'executed_input_area_ratio']})
        info.update(cache_stages=stage_stats, cache_blocks=stats,
                    cache_bytes=sum(v.numel() * v.element_size() for v in values + ages) + info['state_bytes'],
                    block_cache_dense=full, spatial_full_update=self.threshold is None,
                    selector='disabled_full_update' if self.threshold is None else 'aligned_gru_hidden_relative_rms',
                    execution=self.execution)
        return prediction, new, info

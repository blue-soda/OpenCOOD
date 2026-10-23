"""Whole aligned activation caches, with selected spatial-basis mixing.

No teacher values enter recurrence. Reused cells carry their old descriptor
reference so slow change is not hidden by updating the reference every frame.
"""
import math
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.models.sub_modules.int_anchor_residual import refresh_reason, predict


class SpatialBasisCache:
    def __init__(self, stage, bounds, threshold=.5, period=3, execution='selected', dilation=4,
                 confidence_threshold=None, warp=None, cache_dtype=None, max_full_age_s=.25):
        if not isinstance(stage, SpatialBasisStage) or stage.training:
            raise ValueError('Frozen spatial-basis stage required')
        if not math.isfinite(threshold) or threshold < 0 or period not in (1, 2, 3, 4, 5):
            raise ValueError('Finite threshold and period1..5 required')
        if not math.isfinite(max_full_age_s) or not 0 < max_full_age_s <= .45:
            raise ValueError('Full refresh age must be in (0,.45] seconds')
        self.max_full_age_us = round(max_full_age_s*1e6)
        if execution not in ('selected', 'masked_dense') or dilation not in (0, 4):
            raise ValueError('Unsupported execution or change-mask dilation')
        if confidence_threshold is not None and not 0 < confidence_threshold < 1:
            raise ValueError('Confidence threshold must be inside (0,1)')
        self.stage, self.bounds, self.threshold = stage, bounds, threshold
        self.period, self.execution, self.dilation = period, execution, dilation
        self.confidence_threshold = confidence_threshold
        self.warp = warp if warp is not None else warp_bev
        if cache_dtype not in (None, torch.float16):
            raise ValueError('Only original or float16 activation cache storage supported')
        self.cache_dtype = cache_dtype

    def step(self, temporal, state, meta, reset=False):
        if torch.is_grad_enabled() or self.stage.training:
            raise ValueError('Frozen inference required')
        current = self.stage.observe(temporal)
        descriptor = early_descriptor(current, 1)
        stamp = int(meta['timestamp_us'])
        reason = 'reset' if reset or state is None or state['scene'] != meta['segment'] else None
        if reason is None:
            if stamp <= state['stamp']:
                raise ValueError('Noncausal spatial-cache stream')
            if stamp-state['full_stamp'] >= self.max_full_age_us:
                reason = 'age'
            elif state['frames']+1 >= self.period:
                reason = 'period'
        uncovered = 0.; extra_macs = 0; priority_fraction = added_fraction = 0.
        if reason:
            output, maps = self.stage.dense_from_stem(current)
            reference = descriptor
            mask = torch.ones_like(current[:, :1], dtype=torch.bool)
            frames, full_stamp = 0, stamp
        else:
            # A single warp shares geometry/coverage across all four maps and
            # their low-channel reference. Default mode matches mature FM:
            # nearest-neighbor metric SE(2), not optical-flow prediction.
            values = state['maps']+[state['reference']]
            if self.confidence_threshold is not None:
                confidence = state['confidence']
                if confidence.shape != current[:, :1].shape:
                    raise ValueError('Expected previous own detection confidence at the state grid')
                values = values+[confidence]
            # Reference/confidence remain FP32; cat promotes stored activations
            # back to the compute dtype before alignment/convolution. No AMP.
            packed = torch.cat(values, 1).to(current.dtype)
            aligned, valid = self.warp(packed, state['pose'], meta['pose'], self.bounds)
            channels = current.shape[1]
            old = list(aligned[:, :4*channels].split(channels, 1))
            reference = aligned[:, 4*channels:4*channels+descriptor.shape[1]]
            mask = local_change_mask(descriptor, reference, valid, self.threshold)
            if self.dilation:
                mask = F.max_pool2d(mask.float(), 2*self.dilation+1, 1, self.dilation) > 0
            if self.confidence_threshold is not None:
                priority = aligned[:, -1:] >= self.confidence_threshold
                if self.dilation:
                    priority = F.max_pool2d(priority.float(), 2*self.dilation+1, 1, self.dilation) > 0
                priority_fraction = float(priority.float().mean())
                added_fraction = float((priority & ~mask).float().mean())
                mask = mask | priority
            assert bool((mask | valid).all())
            output, maps, extra_macs = self.stage.selected_from_stem(current, old, mask, self.execution)
            reference = torch.where(mask, descriptor, reference)
            uncovered = float((~valid).float().mean())
            frames, full_stamp = state['frames']+1, state['full_stamp']
        count = int(mask.sum()); cells = mask.numel()
        saved_maps = [v.detach().to(dtype=self.cache_dtype or v.dtype) for v in maps]
        new_state = dict(maps=saved_maps, reference=reference.detach(),
            pose=meta['pose'], stamp=stamp, full_stamp=full_stamp, scene=meta['segment'], frames=frames)
        size = sum(v.numel()*v.element_size() for v in new_state['maps']+[new_state['reference']])
        selected_execution = not reason and self.execution == 'selected'
        info = dict(spatial_refresh=reason is not None, spatial_refresh_reason=reason,
            spatial_write_fraction=count/float(cells),
            spatial_mixer_executed_fraction=count/float(cells) if selected_execution else 1.,
            spatial_basis_executed_fraction=0. if getattr(self.stage, 'spatial_basis_fused', False) else (float(count > 0) if selected_execution else 1.),
            spatial_selected_cells=count, spatial_cells=cells, spatial_extra_linear_macs=extra_macs,
            spatial_state_bytes=size, spatial_uncovered_fraction=uncovered,
            spatial_priority_fraction=priority_fraction, spatial_guard_added_fraction=added_fraction,
            spatial_full_age=(stamp-full_stamp)/1e6)
        return output, new_state, info


class SpatialBasisCacheDetector:
    """Same mature downstream/deep cache, replacing only first-stage execution."""
    def __init__(self, model, config, readout, repair, threshold=.5,
                 period=3, execution='selected', deep_period=3, confidence_threshold=None, warp=None, cache_dtype=None,
                 max_full_age_s=.25):
        if model.training or readout.training or repair.training or deep_period not in (1, 3):
            raise ValueError('Frozen detector and deep period1/3 required')
        self.model, self.readout, self.repair = model, readout, repair
        self.bounds = config['model']['args']['lidar_range']; self.deep_period = deep_period
        self.warp = warp if warp is not None else warp_bev
        self.spatial = SpatialBasisCache(model.backbone.resnet.layer0, self.bounds, threshold, period, execution,
                                        confidence_threshold=confidence_threshold, warp=warp, cache_dtype=cache_dtype,
                                        max_full_age_s=max_full_age_s)

    def step(self, data, state, meta):
        m = self.model; p = data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch = {k:p[k] for k in ('voxel_features', 'voxel_coords', 'voxel_num_points')}
            batch['batch_size'] = 1
            spatial = m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx, ny, _ = m.scatter.model_cfg['grid_size']
            spatial = m.cls_head.weight.new_zeros(1, 64, int(ny), int(nx))
        temporal, fm, info = m.feature_memory.step(spatial, None if state is None else state['fm'], meta)
        first, spatial_state, detail = self.spatial.step(temporal,
            None if state is None else state['spatial'], meta, bool(info['reset']))
        current = self.readout.current(m.backbone.deblocks[0](first))
        cache = None if state is None else state['cache']
        reason = refresh_reason(cache, meta, info['reset'], self.deep_period)
        if reason:
            second = m.backbone.resnet.layer1(first); third = m.backbone.resnet.layer2(second)
            deep = torch.cat([m.backbone.deblocks[1](second), m.backbone.deblocks[2](third)], 1)
            projected = self.readout.history(deep)
            cache = dict(value=projected, pose=meta['pose'], stamp=int(meta['timestamp_us']), frames=0)
            uncovered = 0.
        else:
            projected, valid = self.warp(cache['value'], cache['pose'], meta['pose'], self.bounds)
            uncovered = float((~valid).float().mean())
            projected = self.repair(torch.cat([first, projected, valid.to(projected.dtype)], 1))
            cache = dict(cache, frames=cache['frames']+1)
        prediction = predict(m, self.readout.fuse(current, projected))
        if self.spatial.confidence_threshold is not None:
            # Only this mode's previous-frame prediction drives the NEXT mask.
            # No current expensive features, teacher detections, GT or boxes.
            confidence = prediction['psm'].sigmoid().amax(1, keepdim=True).detach()
            if confidence.shape[-2:] != first.shape[-2:]:
                confidence = F.interpolate(confidence, size=first.shape[-2:], mode='nearest')
            spatial_state['confidence'] = confidence
            detail['spatial_state_bytes'] += confidence.numel()*confidence.element_size()
        info.update(detail)
        info.update(local_partial=reason is None, anchor_stamp=cache['stamp'], anchor_depth=cache['frames'],
            uncovered_fraction=uncovered, anchor_bytes=cache['value'].numel()*cache['value'].element_size())
        return prediction, dict(fm=fm, cache=cache, spatial=spatial_state), info

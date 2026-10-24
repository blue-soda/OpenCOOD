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
                 confidence_threshold=None, warp=None, cache_dtype=None, max_full_age_s=.25, cell_age_limit_s=None,
                 allow_cache_grad=False, confidence_dilation=None):
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
        if confidence_dilation not in (None,0,4):
            raise ValueError('Confidence support radius must be0 or4, or follow content dilation')
        self.confidence_dilation = dilation if confidence_dilation is None else confidence_dilation
        self.warp = warp if warp is not None else warp_bev
        if cache_dtype not in (None, torch.float16):
            raise ValueError('Only original or float16 activation cache storage supported')
        self.cache_dtype = cache_dtype
        if cell_age_limit_s not in (None,.25):
            raise ValueError('First cell-age experiment requires a .25 second bound')
        if cell_age_limit_s is not None and period != 3:
            raise ValueError('Cell-age policy replaces the period3 control only')
        self.cell_age_limit_s = cell_age_limit_s
        if allow_cache_grad and (execution != 'masked_dense' or not getattr(stage,'allow_cache_grad',False)):
            raise ValueError('Cache training requires explicitly enabled compact masked-dense stage')
        self.allow_cache_grad = allow_cache_grad
        if getattr(stage,'nested_dependency_masks',False) and (dilation!=0 or cell_age_limit_s is not None):
            raise ValueError('Dependency supports require an undilated core and periodic refresh')

    def step(self, temporal, state, meta, reset=False):
        if (torch.is_grad_enabled() and not self.allow_cache_grad) or self.stage.training:
            raise ValueError('Frozen inference required')
        current = self.stage.observe(temporal)
        descriptor = early_descriptor(current, 1)
        stamp = int(meta['timestamp_us'])
        reason = 'reset' if reset or state is None or state['scene'] != meta['segment'] else None
        if reason is None:
            if stamp <= state['stamp']:
                raise ValueError('Noncausal spatial-cache stream')
            if self.cell_age_limit_s is None and stamp-state['full_stamp'] >= self.max_full_age_us:
                reason = 'age'
            elif self.cell_age_limit_s is None and state['frames']+1 >= self.period:
                reason = 'period'
        uncovered = 0.; extra_macs = 0; priority_fraction = added_fraction = 0.
        age_added = 0.
        if reason:
            output, maps = self.stage.dense_from_stem(current)
            reference = descriptor
            mask = torch.ones_like(current[:, :1], dtype=torch.bool)
            frames, full_stamp = 0, stamp
            if self.cell_age_limit_s is not None: ages = torch.zeros_like(current[:,:1])
        else:
            # A single warp shares geometry/coverage across all four maps and
            # their low-channel reference. Default mode matches mature FM:
            # nearest-neighbor metric SE(2), not optical-flow prediction.
            values = state['maps']+[state['reference']]
            if self.cell_age_limit_s is not None:
                ages = state['ages']
                if ages.shape != current[:,:1].shape or not torch.isfinite(ages).all() or bool((ages<0).any()):
                    raise ValueError('Valid per-cell age map required')
                values = values+[ages]
            if self.confidence_threshold is not None:
                confidence = state['confidence']
                if confidence.shape != current[:, :1].shape:
                    raise ValueError('Expected previous own detection confidence at the state grid')
                values = values+[confidence]
            # Reference/confidence remain FP32; cat promotes stored activations
            # back to the compute dtype before alignment/convolution. No AMP.
            packed = torch.cat(values, 1).to(current.dtype)
            aligned, valid = self.warp(packed, state['pose'], meta['pose'], self.bounds)
            channels = getattr(self.stage,'cache_channels',[current.shape[1]]*4)
            state_channels = sum(channels)
            old = list(aligned[:, :state_channels].split(channels, 1))
            reference = aligned[:, state_channels:state_channels+descriptor.shape[1]]
            mask = local_change_mask(descriptor, reference, valid, self.threshold)
            if self.dilation:
                mask = F.max_pool2d(mask.float(), 2*self.dilation+1, 1, self.dilation) > 0
            if self.confidence_threshold is not None:
                priority = aligned[:, -1:] >= self.confidence_threshold
                if self.confidence_dilation:
                    priority = F.max_pool2d(priority.float(), 2*self.confidence_dilation+1, 1, self.confidence_dilation) > 0
                priority_fraction = float(priority.float().mean())
                added_fraction = float((priority & ~mask).float().mean())
                mask = mask | priority
            if self.cell_age_limit_s is not None:
                offset = state_channels+descriptor.shape[1]
                ages = aligned[:,offset:offset+1]+(stamp-state['stamp'])/1e6
                expired = ages >= self.cell_age_limit_s
                age_added = float((expired & ~mask).float().mean())
                # Age alone does not indicate changed input; no extra halo
                # dilation beyond the unchanged content/priority masks.
                mask = mask | expired
            assert bool((mask | valid).all())
            output, maps, extra_macs = self.stage.selected_from_stem(current, old, mask, self.execution)
            reference = torch.where(mask, descriptor, reference)
            uncovered = float((~valid).float().mean())
            frames, full_stamp = state['frames']+1, state['full_stamp']
            if self.cell_age_limit_s is not None: ages = torch.where(mask,torch.zeros_like(ages),ages)
        count = int(mask.sum()); cells = mask.numel()
        saved_maps = [v.detach().to(dtype=self.cache_dtype or v.dtype) for v in maps]
        new_state = dict(maps=saved_maps, reference=reference.detach(),
            pose=meta['pose'], stamp=stamp, full_stamp=full_stamp, scene=meta['segment'], frames=frames)
        size = sum(v.numel()*v.element_size() for v in new_state['maps']+[new_state['reference']])
        if self.cell_age_limit_s is not None:
            assert bool((ages < self.cell_age_limit_s).all())
            new_state['ages'] = ages.detach()
            size += ages.numel()*ages.element_size()
        selected_execution = not reason and self.execution == 'selected'
        info = dict(spatial_refresh=reason is not None, spatial_refresh_reason=reason,
            spatial_write_fraction=count/float(cells),
            spatial_mixer_executed_fraction=count/float(cells) if selected_execution else 1.,
            spatial_basis_executed_fraction=0. if getattr(self.stage, 'spatial_basis_fused', False) else (float(count > 0) if selected_execution else 1.),
            spatial_selected_cells=count, spatial_cells=cells, spatial_extra_linear_macs=extra_macs,
            spatial_state_bytes=size, spatial_uncovered_fraction=uncovered,
            spatial_priority_fraction=priority_fraction, spatial_guard_added_fraction=added_fraction,
            spatial_full_age=(stamp-full_stamp)/1e6)
        if self.cell_age_limit_s is not None:
            info.update(spatial_max_cell_age_s=float(ages.max()),spatial_age_added_fraction=age_added)
        if getattr(self.stage, 'spatial_input_projected', False):
            info['spatial_basis_executed_fraction'] = 0.
            info['spatial_projection_executed_fraction'] = 1.
            if getattr(self.stage,'compact_projected_state',False) and selected_execution:
                info['spatial_projection_executed_fraction'] = (1.+count/float(cells))/2.
        if getattr(self.stage,'nested_dependency_masks',False):
            layer_cells=[cells]*4 if reason else self.stage.last_layer_cells
            info['spatial_layer_write_fractions']=[n/float(cells) for n in layer_cells]
            if selected_execution:
                kernels=[c for b in self.stage.blocks for c in (b.conv1,b.conv2)]
                info['spatial_mixer_executed_fraction']=sum(n*c.mix.weight.numel() for n,c in zip(layer_cells,kernels))/float(cells*sum(c.mix.weight.numel() for c in kernels))
                info['spatial_projection_executed_fraction']=(2.+(layer_cells[0]+layer_cells[2])/float(cells))/4.
        return output, new_state, info


class SpatialBasisCacheDetector:
    """Same mature downstream/deep cache, replacing only first-stage execution."""
    def __init__(self, model, config, readout, repair, threshold=.5,
                 period=3, execution='selected', deep_period=3, confidence_threshold=None, warp=None, cache_dtype=None,
                 max_full_age_s=.25, cell_age_limit_s=None, spatial_dilation=4, confidence_dilation=None):
        if model.training or readout.training or repair.training or deep_period not in (1, 3):
            raise ValueError('Frozen detector and deep period1/3 required')
        self.model, self.readout, self.repair = model, readout, repair
        self.bounds = config['model']['args']['lidar_range']; self.deep_period = deep_period
        self.warp = warp if warp is not None else warp_bev
        self.spatial = SpatialBasisCache(model.backbone.resnet.layer0, self.bounds, threshold, period, execution,
                                        confidence_threshold=confidence_threshold, warp=warp, cache_dtype=cache_dtype,
                                        max_full_age_s=max_full_age_s,cell_age_limit_s=cell_age_limit_s,dilation=spatial_dilation,
                                        confidence_dilation=confidence_dilation)

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

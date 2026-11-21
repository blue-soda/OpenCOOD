"""Frozen deep-state writeback with the P4 shallow/middle and refresh paths."""
import copy
from contextlib import nullcontext
from functools import partial
from typing import NamedTuple, Optional, Tuple

import torch
from torch import nn

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_mechanism_probes import SingleGateStateUpdate
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_pyramid_cache import PackedPyramidStage
from opencood.models.sub_modules.int_pyramid_regions import (
    coalesced_refresh, plan_coalesced_refresh, plan_counted_host_refresh,
)
from opencood.models.sub_modules.int_two_scale import StateUpdate


class OrdinaryDeepInputs(NamedTuple):
    fine: torch.Tensor
    coarse: torch.Tensor
    old_coarse: torch.Tensor
    aligned: torch.Tensor
    valid: torch.Tensor
    age: torch.Tensor
    dt: float
    deep_route: Optional[Tuple[torch.Tensor, torch.Tensor]]
    coarse_stored: Optional[torch.Tensor] = None


class PackedLightDeepStage(PackedPyramidStage):
    """The learned P4 ordinary deep path with its exact input halo."""

    def __init__(self, encoder, update):
        nn.Module.__init__(self)
        depthwise, activation, projection = encoder
        if (not isinstance(depthwise, nn.Conv2d) or not isinstance(activation, nn.ReLU)
                or not isinstance(projection, nn.Conv2d) or not isinstance(update, SingleGateStateUpdate)
                or not isinstance(update.candidate, StateUpdate)):
            raise ValueError('Expected the ordinary P4 deep encoder and gated state update')
        if ((depthwise.kernel_size, depthwise.stride, depthwise.padding, depthwise.dilation)
                != ((3, 3), (2, 2), (1, 1), (1, 1))
                or (projection.kernel_size, projection.stride, projection.padding, projection.dilation)
                != ((1, 1), (1, 1), (0, 0), (1, 1))):
            raise ValueError('Unsupported convolution geometry for the light deep halo')
        self.encoder, self.update = encoder, update
        self.input_channels = depthwise.in_channels
        self.output_channels = projection.out_channels
        # One input cell for the observation, two output cells for StateUpdate.
        self.radius = 1 + 2 * update.candidate.radius


class DeepBlockCache:
    def __init__(self, source, tile, operator='heavy', deep_interface=None,
                 planner=plan_coalesced_refresh, phase=nullcontext, crop_first=False, counted_host=False,
                 deep_projector=None, coarse_interface=None):
        if source.training or not source.light_observations or tile < 1:
            raise ValueError('Frozen light-observation source and positive tile size required')
        self.source = source
        self.core = source.core
        self.tile = tile
        self.operator = operator
        # Optional mapping from the computed state and written mask to stored and readout values.
        self.deep_interface = deep_interface
        self.coarse_interface = coarse_interface
        self.planner = planner
        self.phase = phase
        self.crop_first = crop_first
        self.counted_host = counted_host
        if (deep_projector is not None and deep_interface is not None
                and not getattr(deep_projector, 'accepts_quantized_readout', False)):
            raise ValueError('Test the projection cache with unmodified float task features')
        self.deep_projector = deep_projector
        if operator == 'heavy':
            self.stage = PackedPyramidStage(self.core.deep_encoder, self.core.refresh_updates['deep']).eval()
        elif operator == 'light':
            self.stage = PackedLightDeepStage(self.core.deep_light, self.core.deep_update).eval()
        else:
            raise ValueError('Unknown ordinary deep operator')

    def refresh_reason(self, state, meta):
        reset, _ = reset_reason(state, meta)
        if reset:
            return reset
        if state['since_refresh'] + 1 >= self.source.period:
            return 'period'
        if (int(meta['timestamp_us']) - state['refresh_timestamp_us']) / 1e6 >= self.source.max_age - 1e-6:
            return 'age'
        return None

    def prepare(self, spatial, state, meta):
        if torch.is_grad_enabled() or spatial.shape[0] != 1 or self.refresh_reason(state, meta):
            raise ValueError('Expected a frozen ordinary frame with one causal stream')
        _, dt = reset_reason(state, meta)
        current = spatial
        features = []
        coarse_stored = None
        for key, encoder, update, stride in (
            ('value', self.core.observation, self.core.fine_update, 2),
            ('coarse', self.core.coarse_light, self.core.coarse_update, 4),
        ):
            with self.phase('fine_coarse_compute'):
                encoded = encoder(current)
            with self.phase('fine_coarse_warp'):
                bounds = block_lidar_range(self.source.bounds, spatial.shape[-2:], encoded.shape[-2:], stride)
                old, valid = warp_bev(state[key], state['pose'], meta['pose'], bounds)
            with self.phase('fine_coarse_compute'):
                update.delta_t = dt
                current = update(encoded, old, valid)
            features.append(current)
        with self.phase('deep_warp'):
            bounds = block_lidar_range(self.source.bounds, spatial.shape[-2:], state['deep'].shape[-2:], 8)
            age = state.get('deep_age')
            if age is None:
                elapsed = (state['timestamp_us'] - state['refresh_timestamp_us']) / 1e6
                age = torch.full_like(state['deep'][:, :1], elapsed)
            value = torch.cat((state['deep'], age), 1)
            route = None
            if self.deep_projector is not None and self.deep_projector.indexed:
                aligned, valid, route = self.deep_projector.align_task(value, state['pose'], meta['pose'], bounds)
            else:
                aligned, valid = warp_bev(value, state['pose'], meta['pose'], bounds)
            return OrdinaryDeepInputs(features[0], features[1], old, aligned[:, :-1], valid,
                                      aligned[:, -1:] + dt, dt, route, coarse_stored)

    def mandatory(self, frame):
        required = ~frame.valid | (frame.age >= self.source.max_age - 1e-6)
        return expand_tiles(tile_mask(required, self.tile), self.tile, required.shape[-2:])

    def predict(self, frame, deep, deep_projection=None):
        projected = [self.core.deblocks[0](frame.fine), self.core.deblocks[1](frame.coarse)]
        projected.append(self.core.deblocks[2](deep) if deep_projection is None else deep_projection)
        features = self.core.readout(torch.cat(projected, 1))
        output = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None:
            output['dm'] = self.core.dir_head(features)
        return output

    def pack(self, frame):
        self.stage.update.delta_t = frame.dt
        return self.stage.pack(frame.coarse, frame.aligned, frame.valid)

    def step_spatial(self, spatial, state, meta, requests=None, dense_reference=False):
        if self.refresh_reason(state, meta):
            kwargs = {} if self.deep_interface is None else {'deep_interface': self.deep_interface}
            if self.coarse_interface is not None:
                kwargs['coarse_interface'] = self.coarse_interface
            if self.phase is not nullcontext:
                kwargs['phase'] = self.phase
            if self.deep_projector is not None:
                kwargs['deep_projector'] = self.deep_projector
            prediction, new, info = self.source.step_spatial(spatial, state, meta, **kwargs)
            new['deep_age'] = torch.zeros_like(new['deep'][:, :1])
            return prediction, new, dict(info, stage_conv_ratio=1., actual_write_fraction=1., fallback=False)
        frame = self.prepare(spatial, state, meta)
        with self.phase('mask_guard'):
            mask = self.mandatory(frame)
            if callable(requests):
                requests = requests(frame, mask)
            if requests is not None:
                if requests.dtype != torch.bool or requests.shape != mask.shape:
                    raise ValueError('Expected a boolean mask on the deep lattice')
                mask = mask | requests
            if not self.counted_host:
                needs_work = bool(mask.any())
        counted_plan = None
        if self.counted_host:
            with self.phase('region_plan'):
                counted_plan = plan_counted_host_refresh(mask, frame.coarse.shape[-2:], self.stage.radius, self.tile)
                needs_work = counted_plan is not None
        if needs_work:
            with self.phase('state_pack'):
                if self.crop_first:
                    self.stage.update.delta_t = frame.dt
                    packed = frame.coarse
                    region_inputs = partial(self.stage.region_inputs, frame.coarse, frame.aligned, frame.valid)
                else:
                    packed = self.pack(frame)
                    region_inputs = None
            deep, written, info = coalesced_refresh(
                self.stage, packed, frame.aligned, mask, tile=self.tile, dense_reference=dense_reference,
                planner=self.planner, phase=self.phase, region_inputs=region_inputs, counted_plan=counted_plan,
            )
        else:
            deep, written = frame.aligned, mask
            info = dict(stage_conv_ratio=0., actual_write_fraction=0., fallback=False)
        with self.phase('state_commit'):
            stored, sent = (deep, deep) if self.deep_interface is None else self.deep_interface(deep, written)
            coarse_state = frame.coarse if frame.coarse_stored is None else frame.coarse_stored
            new = dict(value=frame.fine, coarse=coarse_state, deep=stored,
                       deep_age=torch.where(written, torch.zeros_like(frame.age), frame.age),
                       pose=copy.deepcopy(meta['pose']), scene=meta['segment'], timestamp_us=int(meta['timestamp_us']),
                       refresh_timestamp_us=state['refresh_timestamp_us'], since_refresh=state['since_refresh'] + 1)
        with self.phase('readout'):
            projected = None
            if self.deep_projector is not None:
                bounds = block_lidar_range(self.source.bounds, spatial.shape[-2:], sent.shape[-2:], 8)
                projected, new['deep_projection'] = self.deep_projector.advance(
                    sent, written, state['deep_projection'], state['pose'], meta['pose'], bounds, frame.deep_route)
            return self.predict(frame, sent, projected), new, dict(info, reset=None, full_refresh=None)

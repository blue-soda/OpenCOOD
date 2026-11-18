"""Frozen deep-state writeback with the P4 shallow/middle and refresh paths."""
import copy
from typing import NamedTuple

import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_pyramid_cache import PackedPyramidStage
from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh


class OrdinaryDeepInputs(NamedTuple):
    fine: torch.Tensor
    coarse: torch.Tensor
    old_coarse: torch.Tensor
    aligned: torch.Tensor
    valid: torch.Tensor
    age: torch.Tensor


class DeepBlockCache:
    def __init__(self, source, tile):
        if source.training or not source.light_observations or tile < 1:
            raise ValueError('Frozen light-observation source and positive tile size required')
        self.source = source
        self.core = source.core
        self.tile = tile
        self.heavy = PackedPyramidStage(self.core.deep_encoder, self.core.refresh_updates['deep']).eval()

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
        for key, encoder, update, stride in (
            ('value', self.core.observation, self.core.fine_update, 2),
            ('coarse', self.core.coarse_light, self.core.coarse_update, 4),
        ):
            encoded = encoder(current)
            bounds = block_lidar_range(self.source.bounds, spatial.shape[-2:], encoded.shape[-2:], stride)
            old, valid = warp_bev(state[key], state['pose'], meta['pose'], bounds)
            update.delta_t = dt
            current = update(encoded, old, valid)
            features.append(current)
        bounds = block_lidar_range(self.source.bounds, spatial.shape[-2:], state['deep'].shape[-2:], 8)
        age = state.get('deep_age')
        if age is None:
            elapsed = (state['timestamp_us'] - state['refresh_timestamp_us']) / 1e6
            age = torch.full_like(state['deep'][:, :1], elapsed)
        aligned, valid = warp_bev(torch.cat((state['deep'], age), 1), state['pose'], meta['pose'], bounds)
        return OrdinaryDeepInputs(features[0], features[1], old, aligned[:, :-1], valid, aligned[:, -1:] + dt)

    def mandatory(self, frame):
        required = ~frame.valid | (frame.age >= self.source.max_age - 1e-6)
        return expand_tiles(tile_mask(required, self.tile), self.tile, required.shape[-2:])

    def predict(self, frame, deep):
        projected = [layer(value) for layer, value in zip(self.core.deblocks, (frame.fine, frame.coarse, deep))]
        features = self.core.readout(torch.cat(projected, 1))
        output = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None:
            output['dm'] = self.core.dir_head(features)
        return output

    def step_spatial(self, spatial, state, meta, requests=None, dense_reference=False):
        if self.refresh_reason(state, meta):
            prediction, new, info = self.source.step_spatial(spatial, state, meta)
            new['deep_age'] = torch.zeros_like(new['deep'][:, :1])
            return prediction, new, dict(info, stage_conv_ratio=1., actual_write_fraction=1., fallback=False)
        frame = self.prepare(spatial, state, meta)
        mask = self.mandatory(frame)
        if requests is not None:
            if requests.dtype != torch.bool or requests.shape != mask.shape:
                raise ValueError('Expected a boolean mask on the deep lattice')
            mask = mask | requests
        if bool(mask.any()):
            deep, written, info = coalesced_refresh(
                self.heavy, self.heavy.pack(frame.coarse, frame.aligned, frame.valid),
                frame.aligned, mask, tile=self.tile, dense_reference=dense_reference,
            )
        else:
            deep, written = frame.aligned, mask
            info = dict(stage_conv_ratio=0., actual_write_fraction=0., fallback=False)
        new = dict(value=frame.fine, coarse=frame.coarse, deep=deep,
                   deep_age=torch.where(written, torch.zeros_like(frame.age), frame.age),
                   pose=copy.deepcopy(meta['pose']), scene=meta['segment'], timestamp_us=int(meta['timestamp_us']),
                   refresh_timestamp_us=state['refresh_timestamp_us'], since_refresh=state['since_refresh'] + 1)
        return self.predict(frame, deep), new, dict(info, reset=None, full_refresh=None)

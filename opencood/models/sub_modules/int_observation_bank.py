"""Causal three-observation BEV fusion before the complete spatial encoder."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev


class ObservationBankDetector(PyramidStateDetector):
    version = 'observation_bank_three_frame_v1'
    state_keys = ('value', 'older')
    max_history_age = .25

    def __init__(self, teacher, config, memory_policy='recurrent'):
        super().__init__(teacher, config, memory_policy)
        for name in ('fine_update', 'coarse_update', 'deep_update'):
            delattr(self.core, name)
        self.core.evidence_fusion = nn.Sequential(
            nn.Conv2d(196, 128, 1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 64, 1))
        nn.init.zeros_(self.core.evidence_fusion[-1].weight)
        nn.init.zeros_(self.core.evidence_fusion[-1].bias)
        self.train(False)

    def updates_are_initial_identity(self):
        return all(torch.count_nonzero(p).item() == 0
                   for p in (self.core.evidence_fusion[-1].weight,
                             self.core.evidence_fusion[-1].bias))

    def aligned_history(self, spatial, state, meta, reason):
        values, masks, ages = [], [], []
        used = 0
        for key, time_key, pose_key in (
                ('value', 'timestamp_us', 'pose'),
                ('older', 'older_timestamp_us', 'older_pose')):
            old = torch.zeros_like(spatial)
            valid = torch.zeros_like(spatial[:, :1], dtype=torch.bool)
            age_map = torch.zeros_like(spatial[:, :1])
            if reason is None and state[time_key] is not None:
                age = (int(meta['timestamp_us']) - state[time_key]) / 1e6
                if age <= 0:
                    raise ValueError('Noncausal observation bank')
                if age <= self.max_history_age:
                    old, valid = warp_bev(state[key], state[pose_key], meta['pose'], self.bounds)
                    age_map = valid.to(spatial.dtype) * (age / self.max_history_age)
                    used += 1
            values.append(old); masks.append(valid.to(spatial.dtype)); ages.append(age_map)
        return values, masks, ages, used

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
                if gap > self.max_history_age:
                    reason = 'gap'
        old, valid, age, used = self.aligned_history(spatial, state, meta, reason)
        x = spatial
        if not self.spatial_only:
            x = x + self.core.evidence_fusion(torch.cat([spatial] + old + valid + age, 1))
        values = []
        for encoder in (self.core.observation, self.core.coarse_encoder, self.core.deep_encoder):
            x = encoder(x); values.append(x)
        ups = [deblock(value) for deblock, value in zip(self.core.deblocks, values)]
        if len({tuple(x.shape[-2:]) for x in ups}) != 1:
            raise ValueError('Input lattice does not align the mature pyramid readout')
        features = self.core.readout(torch.cat(ups, 1))
        prediction = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None:
            prediction['dm'] = self.core.dir_head(features)
        # Store raw frozen observations, never fused outputs or semantic states.
        new = dict(value=spatial.detach().clone(),
                   older=state['value'] if reason is None else torch.zeros_like(spatial),
                   pose=copy.deepcopy(meta['pose']), scene=meta['segment'],
                   timestamp_us=int(meta['timestamp_us']),
                   older_pose=state['pose'] if reason is None else None,
                   older_timestamp_us=state['timestamp_us'] if reason is None else None)
        size = sum(new[key].numel() * new[key].element_size() for key in self.state_keys)
        return prediction, new, dict(reset=reason, state_bytes=size, cache_bytes=size,
            history_observations_used=used,
            cache_stages=[dict(actual_write_fraction=1., executed_input_area_ratio=1.)])

"""Protected fine refresh plus cheap current observation on ordinary frames."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector, reset_reason
from opencood.models.sub_modules.int_mechanism_probes import SingleGateStateUpdate
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class FineRefreshDetector(PyramidStateDetector):
    """Shares one task state per scale; new ordinary fine path requires training.

    Full-refresh functions are copied from a trained protected detector. They use
    the student's own history, so preserving these functions does not promise
    identical predictions after different ordinary-frame trajectories.
    """
    def __init__(self, source, light_fine=True, period=None):
        nn.Module.__init__(self)
        if type(source) is not ProtectedMultiscaleRefreshDetector:
            raise ValueError('Trained protected detector required')
        self.core = copy.deepcopy(source.core)
        self.pillar_vfe, self.scatter = source.pillar_vfe, source.scatter
        self.bounds = copy.deepcopy(source.bounds)
        self.period = source.period if period is None else period
        if type(self.period) is not int or not 1 <= self.period <= 6:
            raise ValueError('Invalid period')
        self.max_age = source.max_age
        self.light_observations = source.light_observations
        self.light_fine = bool(light_fine)
        self.memory_policy = 'recurrent'
        self.spatial_only = False
        self.variant = 'protected_fine_light' if light_fine else 'protected_fine_dense'
        self.core.refresh_updates['value'] = copy.deepcopy(self.core.fine_update)
        self.core.fine_update = SingleGateStateUpdate(self.core.fine_update, 128)
        self.core.fine_light = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64), nn.ReLU(),
            nn.Conv2d(64, 128, 1))
        self.train(False)

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if spatial.shape[0] != 1 or spatial.shape[1] != 64 or requests is not None:
            raise ValueError('Single64-channel stream without spatial masks required')
        reason, dt = reset_reason(state, meta)
        stamp = int(meta['timestamp_us'])
        refresh = reason
        if refresh is None:
            if state['since_refresh'] + 1 >= self.period:
                refresh = 'period'
            elif (stamp - state['refresh_timestamp_us']) / 1e6 >= self.max_age - 1e-6:
                refresh = 'age'
        values = []
        x = spatial
        for key, encoder, light, update, stride, enabled in (
            ('value', self.core.observation, self.core.fine_light, self.core.fine_update, 2, self.light_fine),
            ('coarse', self.core.coarse_encoder, self.core.coarse_light, self.core.coarse_update, 4, self.light_observations),
            ('deep', self.core.deep_encoder, self.core.deep_light, self.core.deep_update, 8, self.light_observations)):
            current = (encoder if refresh or not enabled else light)(x)
            if reason:
                old = torch.zeros_like(current)
                valid = torch.zeros_like(current[:, :1], dtype=torch.bool)
            else:
                bounds = block_lidar_range(self.bounds, spatial.shape[-2:], current.shape[-2:], stride)
                old, valid = warp_bev(state[key], state['pose'], meta['pose'], bounds)
            if refresh:
                update = self.core.refresh_updates[key]
            update.delta_t = dt
            x = update(current, old, valid)
            values.append(x)
        features = self.core.readout(torch.cat([m(v) for m, v in zip(self.core.deblocks, values)], 1))
        pred = dict(psm=self.core.cls_head(features), rm=self.core.reg_head(features))
        if self.core.dir_head is not None:
            pred['dm'] = self.core.dir_head(features)
        new = {k: v.detach() if detach_state else v for k, v in zip(self.state_keys, values)}
        new.update(pose=copy.deepcopy(meta['pose']), scene=meta['segment'], timestamp_us=stamp,
                   since_refresh=0 if refresh else state['since_refresh'] + 1,
                   refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        return pred, new, dict(reset=reason, full_refresh=refresh,
            cache_bytes=sum(new[k].numel() * new[k].element_size() for k in self.state_keys),
            fine_encoder_executed=bool(refresh) or not self.light_fine,
            middle_deep_encoder_executed=bool(refresh) or not self.light_observations,
            state_updates_executed=True)

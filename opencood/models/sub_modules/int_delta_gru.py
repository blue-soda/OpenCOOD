"""Dense reference for innovation accumulation and event-refreshed readout.

No sparse kernel or speedup is claimed. Cache is [U, R, Y, age, mask, valid].
U accumulates hidden prediction error, R holds emitted evidence, Y is readout.
Fresh/expired cells receive analog bootstrap; subsequent updates use events.
"""
import math
import torch
import torch.nn.functional as F

from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory, INTFusionModule
from opencood.models.sub_modules.int_spiking_gru import signed_event


def support_rates(mask):
    return {'active_cells': float(mask.float().mean()),
            'active_tiles8': float(F.max_pool2d(mask.float(), 8, 8, ceil_mode=True).mean())}


class INTDeltaGRUFusion(INTFusionModule):
    def __init__(self, config, threshold=.25, refresh_s=1.):
        super().__init__(config)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('threshold must be positive and finite')
        if not math.isfinite(refresh_s) or refresh_s <= 0:
            raise ValueError('refresh_s must be positive and finite')
        self.channels = config['input_channels']
        self.state_channels = 3*self.channels+3
        self.threshold, self.refresh_s = float(threshold), float(refresh_s)
        self.last_diagnostics = {}

    def forward(self, x, past_x_list, delta_t=0., detach_state=True):
        u, memory, readout, age, old_mask, valid = torch.split(
            past_x_list[0], [self.channels]*3+[1]*3, dim=1)
        age = age + float(delta_t)
        force = (valid < .5) | (age >= self.refresh_s)
        mask = (x.abs().sum(1, keepdim=True) > 0).to(x.dtype)
        reset = self.reset_gate_conv(torch.cat([x, memory], 1))
        update = self.update_gate_conv(torch.cat([x, memory], 1))
        candidate = self.candidate_conv(torch.cat([x, reset*memory], 1))
        target = update*memory + (1-update)*candidate
        voltage = u + (target-memory)
        spikes = signed_event(voltage, self.threshold)
        # Bootstrap and periodic refresh are explicitly analog, not spikes.
        spikes = torch.where(force, torch.zeros_like(spikes), spikes)
        memory_next = torch.where(force, target, memory+self.threshold*spikes)
        u_next = torch.where(force, torch.zeros_like(u), voltage-self.threshold*spikes)
        age_next = torch.where(force, torch.zeros_like(age), age)
        dirty = force | (spikes != 0).any(1, keepdim=True) | (mask != old_mask)
        # A 3x3 readout convolution needs a one-cell output halo.
        refresh = F.max_pool2d(dirty.to(x.dtype), 3, 1, 1) > 0
        dense = self.post_fusion_conv(torch.cat([memory_next, age_next, mask], 1))
        output = torch.where(refresh, dense, readout)
        cached = torch.cat([u_next, memory_next, output, age_next, mask,
                            torch.ones_like(valid)], 1)
        with torch.no_grad():
            if not torch.isfinite(cached).all():
                raise RuntimeError('Nonfinite innovation state')
            emitted = spikes != 0
            self.last_diagnostics = dict(
                emission_nonzero_fraction=float(emitted.float().mean()),
                emission_positive_fraction=float((spikes>0).float().mean()),
                emission_negative_fraction=float((spikes<0).float().mean()),
                emission_on_observed_fraction=float((emitted & (mask>0)).sum()) /
                    max(1., float(mask.sum())*self.channels),
                membrane_abs_max=float(u_next.abs().max()),
                forced_refresh_fraction=float(force.float().mean()),
                readout_refresh_fraction=float(refresh.float().mean()),
                readout_refresh_tiles8=support_rates(refresh)['active_tiles8'],
                event_cells_fraction=float(emitted.any(1,keepdim=True).float().mean()),
                dense_execution=True)
        return output, cached.detach() if detach_state else cached


class INTDeltaGRUMemory(INTFeatureMemory):
    def __init__(self, channels, lidar_range, threshold=.25, refresh_s=1., max_gap_s=.25):
        super().__init__('gru', channels, lidar_range, max_gap_s=max_gap_s,
                         fusion_factory=lambda cfg: INTDeltaGRUFusion(cfg, threshold, refresh_s))
        self.mode = 'delta_gru'

    def step(self, current, state, meta, reset=False, align=True, detach_state=True):
        output, state, info = super().step(current,state,meta,reset,align,detach_state)
        info.update(self.fusion.last_diagnostics)
        return output,state,info

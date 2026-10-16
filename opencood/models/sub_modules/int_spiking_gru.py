"""GRU layer/parameter/cache-matched hybrid spiking memory.

All convolutions, BN layers, sigmoid gates and tanh candidate are inherited from
INT's GRU. Only hidden-state emission/reset changes. The readout receives signed
spikes, plus the original age and occupancy auxiliaries; it never reads membrane.
"""
import math

import torch

from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory, INTFusionModule
from opencood.models.sub_modules.int_spike_memory import SurrogateSpike


def signed_event(voltage, threshold):
    """Two binary polarities encoded as -1/0/+1, with surrogate gradients."""
    return (SurrogateSpike.apply(voltage / threshold - 1.) -
            SurrogateSpike.apply(-voltage / threshold - 1.))


class INTSpikingGRUFusion(INTFusionModule):
    def __init__(self, config, threshold=.25):
        if config['type'] != 'infinite_GRU':
            raise ValueError('Spiking GRU requires the original GRU layer layout')
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Fixed threshold must be in (0,1) for the bounded tanh drive')
        super().__init__(config)
        self.threshold = float(threshold)  # Hyperparameter, not a parameter or cached tensor.
        self.last_diagnostics = {}

    def forward(self, x, past_x_list, delta_t=.05, detach_state=True):
        if len(past_x_list) != 1:
            raise ValueError('Exactly one aligned recurrent state is required')
        past = past_x_list[0]
        membrane, age = past[:, :-1], past[:, -1].clone()
        # Retain the exact ANN age rule, including its sum-based nonzero test.
        age[membrane.sum(dim=1) != 0] += float(delta_t)
        reset_gate = self.reset_gate_conv(torch.cat([x, membrane], dim=1))
        update_gate = self.update_gate_conv(torch.cat([x, membrane], dim=1))
        candidate = self.candidate_conv(torch.cat([x, reset_gate * membrane], dim=1))
        voltage = update_gate * membrane + (1 - update_gate) * candidate
        spikes = signed_event(voltage, self.threshold)
        next_membrane = voltage - self.threshold * spikes
        cached = torch.cat([next_membrane, age.unsqueeze(1)], dim=1)
        with torch.no_grad():
            if not torch.isfinite(cached).all():
                raise RuntimeError('Nonfinite spiking GRU state')
            mask = (x.sum(dim=1) != 0).to(x.dtype).unsqueeze(1)
            observed = x.abs().sum(dim=1, keepdim=True) > 0
            emitted = spikes != 0
            self.last_diagnostics = dict(
                emission_nonzero_fraction=float(emitted.float().mean()),
                emission_positive_fraction=float((spikes > 0).float().mean()),
                emission_negative_fraction=float((spikes < 0).float().mean()),
                emission_on_observed_fraction=float((emitted & observed).sum()) /
                    max(1., float(observed.sum()) * spikes.shape[1]),
                membrane_abs_max=float(next_membrane.abs().max()),
                spike_threshold=self.threshold, signed_events=True)
            if getattr(self, 'profile_support', False):
                from opencood.models.sub_modules.int_delta_gru import support_rates
                cells = emitted.any(1, keepdim=True)
                self.last_diagnostics.update(support_rates(cells))
        # Fixed event amplitude preserves the candidate's numerical scale.
        output = self.post_fusion_conv(torch.cat(
            [self.threshold * spikes, age.unsqueeze(1), mask], dim=1))
        return output, cached.detach() if detach_state else cached


class INTSpikingGRUMemory(INTFeatureMemory):
    def __init__(self, channels, lidar_range, threshold=.25, max_gap_s=.25):
        super().__init__('gru', channels, lidar_range, max_gap_s=max_gap_s,
                         fusion_factory=lambda config: INTSpikingGRUFusion(config, threshold))
        self.mode = 'spike_gru'

    def step(self, current, state, meta, reset=False, align=True, detach_state=True):
        output, state, info = super().step(current, state, meta, reset, align, detach_state)
        info.update(self.fusion.last_diagnostics)
        return output, state, info

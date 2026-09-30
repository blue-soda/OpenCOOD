"""Aligned real-frame LIF memory and a matched continuous leaky control.

Only the filtered emission trace is exposed to the detector. Both membrane and
trace are continuous state and are counted in the reported cache footprint.
One LiDAR observation is one update; there are no artificial within-frame steps.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.sub_modules.int_feature_memory import warp_bev


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        # Derivative of atan(pi*x)/pi; forward remains exactly binary.
        return gradient / (1 + (math.pi * value).square())


def positive_parameter(value, channels):
    if value <= .001:
        raise ValueError('Initial time constants/threshold must exceed .001')
    return nn.Parameter(torch.full((1, channels, 1, 1), math.log(math.expm1(value - .001))))


class INTSpikeMemory(nn.Module):
    def __init__(self, mode, channels, lidar_range, memory_channels=32,
                 tau_u=.2, tau_r=.1, threshold=1., input_gain=.25, max_gap_s=.25):
        super().__init__()
        if mode not in ('lif', 'leaky') or memory_channels < 1:
            raise ValueError('Expected lif/leaky and positive memory_channels')
        self.mode, self.channels = mode, memory_channels
        self.lidar_range, self.max_gap_s = lidar_range, max_gap_s
        self.input_conv = nn.Conv2d(channels, memory_channels, 3, padding=1, bias=False)
        self.readout = nn.Conv2d(memory_channels, channels, 1, bias=False)
        self.raw_tau_u = positive_parameter(tau_u, memory_channels)
        self.raw_tau_r = positive_parameter(tau_r, memory_channels)
        self.raw_threshold = positive_parameter(threshold, memory_channels)
        nn.init.dirac_(self.input_conv.weight)
        nn.init.dirac_(self.readout.weight)
        with torch.no_grad():
            self.input_conv.weight.mul_(input_gain)
            self.readout.weight.zero_()
            for channel in range(channels):
                self.readout.weight[channel, channel % memory_channels, 0, 0] = 1.

    def dynamics(self, current, past, delta_t):
        u, trace = past.split(self.channels, dim=1)
        tau_u = F.softplus(self.raw_tau_u) + .001
        tau_r = F.softplus(self.raw_tau_r) + .001
        threshold = F.softplus(self.raw_threshold) + .001
        voltage = torch.exp(-float(delta_t) / tau_u) * u + self.input_conv(current)
        if self.mode == 'lif':
            emission = SurrogateSpike.apply(voltage / threshold - 1.)
            membrane = voltage - threshold * emission  # differentiable soft reset
        else:
            emission = F.relu(voltage / threshold)
            membrane = voltage  # no threshold event or reset in the ANN control
        trace = torch.exp(-float(delta_t) / tau_r) * trace + emission
        # Linear readout permits signed features and preserves surrogate gradients
        # at an all-zero trace (an extra ReLU would block them at zero).
        output = self.readout(trace)
        return output, torch.cat([membrane, trace], dim=1), emission

    def step(self, current, state, meta, reset=False, align=True, detach_state=True):
        if current.shape[0] != 1:
            raise ValueError('Use an explicit state per sequence; batch must be 1')
        reason, delta_t = ('initial' if state is None else None), 0.
        if state is not None and not reset:
            if state['scene'] != meta['segment']:
                reason = 'scene'
            else:
                delta_t = (int(meta['timestamp_us']) - state['timestamp_us']) / 1e6
                if delta_t <= 0:
                    raise ValueError('Non-causal or repeated frame in one stream')
                if delta_t > self.max_gap_s:
                    reason = 'gap'
        if reset:
            reason = 'explicit'
        if reason is not None:
            past = current.new_zeros((1, 2*self.channels, *current.shape[2:]))
            coverage = 0.
        elif align:
            past, valid = warp_bev(state['value'], state['pose'], meta['pose'], self.lidar_range)
            coverage = float(valid.float().mean())
        else:
            past, coverage = state['value'], 1.
        output, value, emission = self.dynamics(current, past, delta_t)
        with torch.no_grad():
            if not torch.isfinite(value).all():
                raise RuntimeError('Non-finite temporal state')
            active = current.abs().sum(1, keepdim=True) > 0
            emitted = emission > 0
            info = dict(reset=reason, delta_t=delta_t, coverage=coverage,
                        state_bytes=value.numel()*value.element_size(),
                        warp=reason is None and align,
                        emission_nonzero_fraction=float(emitted.float().mean()),
                        emission_mean=float(emission.mean()),
                        emission_on_observed_fraction=float((emitted & active).sum()) /
                        max(1., float(active.sum()) * self.channels),
                        membrane_abs_max=float(value[:, :self.channels].abs().max()),
                        trace_abs_max=float(value[:, self.channels:].abs().max()),
                        binary_emission=self.mode == 'lif')
        new_state = dict(value=value.detach() if detach_state else value,
                         scene=meta['segment'], pose=meta['pose'], timestamp_us=int(meta['timestamp_us']))
        return output, new_state, info

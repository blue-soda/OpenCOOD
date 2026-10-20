"""Constant-drive IF stream reference, independent of transport packet count.

Analytical threshold crossings t=n/drive (unit threshold, horizon=1).
Repeated addresses are valid spikes, not duplicate messages. CPU scheduling
is a correctness reference, NOT a neuromorphic speed/energy implementation.
"""
from dataclasses import dataclass
import numpy as np
import torch


class _HardCounts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, hard):
        return hard.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


@dataclass
class IFStream:
    drive: torch.Tensor
    addresses: np.ndarray
    times: np.ndarray
    max_spikes: int

    def prefix(self, count):
        if type(count) is not int or not 0 <= count <= len(self.addresses):
            raise ValueError('Invalid event prefix length')
        hard = np.bincount(self.addresses[:count].astype(np.int64),
                           minlength=self.drive.numel()).reshape(self.drive.shape)
        hard = self.drive.new_tensor(hard)
        # Fixed-frontier STE: biased, does not differentiate discrete sorting
        # or serialized bytes. Explicitly no gradient through the frontier.
        if count == 0:
            return self.drive * 0
        time = 1. if count == len(self.addresses) else float(self.times[count - 1])
        soft = self.drive * time
        return _HardCounts.apply(soft, hard)


def integrate_and_fire(drive, max_spikes=8):
    """One finite nonnegative CZYX frame; clamp is a declared safety ceiling."""
    if (drive.ndim != 5 or drive.shape[0] != 1 or
            any(n < 1 for n in drive.shape) or not torch.isfinite(drive).all() or
            torch.any(drive < 0) or drive.numel() >= 2**32):
        raise ValueError('Expected finite nonnegative single-frame drive')
    if type(max_spikes) is not int or not 1 <= max_spikes <= 64:
        raise ValueError('Invalid spike ceiling')
    bounded = drive.clamp(max=max_spikes)
    flat = bounded.detach().double().cpu().numpy().ravel()
    addresses, times = [], []
    for ordinal in range(1, max_spikes + 1):
        active = np.flatnonzero(flat >= ordinal)
        addresses.append(active)
        times.append(ordinal / flat[active])
    addresses, times = np.concatenate(addresses), np.concatenate(times)
    order = np.lexsort((addresses, times))
    return IFStream(bounded, addresses[order].astype('<u4'), times[order], max_spikes)


def stream_readout(membrane, counts, pose, contract, receiver, decode):
    """Canonical cumulative-current readout using the existing joint weights.

All integer events are accumulated in source coordinates before one warp.
This equals the retained-residual/fixed-gain receiver mathematically; it
avoids regrouping float additions at transport boundaries. Not an incremental
compute optimization, and no new receiver accuracy mechanism is claimed.
"""
    from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
    from opencood.models.fuse_modules.e3dsnn_progressive import count_readout, check_transform
    check_transform(pose)
    if (counts.shape != (1, receiver.latent_channels) + contract.shape_zyx or
            membrane.shape != (1, receiver.channels) + contract.shape_zyx or
            not torch.isfinite(counts).all() or torch.any(counts < 0) or
            not torch.isfinite(membrane).all()):
        raise ValueError('Wrong receiver shape or nonfinite state')
    if not receiver.retain_residual or not receiver.use_local_membrane:
        raise ValueError('Reference requires the full retained-membrane receiver')
    aligned, coverage = warp_volume(counts, pose, contract.origin_xyz,
                                    contract.step_xyz, geometry_dtype=torch.float64)
    gain = 2 * torch.sigmoid(receiver.local_gate(membrane)) * receiver.current_scale
    current = decode(aligned) * gain
    supported = (aligned.detach() != 0).any(1, keepdim=True) & (coverage > 0)
    return torch.where(supported, count_readout(membrane + current),
                       count_readout(membrane))

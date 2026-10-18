"""Receiver-derived support of nonzero decoded rows, not visibility/free space."""
import torch
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


def message_support(road, transform, coverage, origin, step, mode):
    if mode == 'geometry':
        return coverage
    if mode != 'received':
        raise ValueError('Unknown message support')
    # All-zero rows are omitted by both wire codecs. Derive this after decoding,
    # before interpolation; a fractional result denotes interpolation support.
    occupied = road.detach().ne(0).any(dim=1, keepdim=True).to(road.dtype)
    received, _ = warp_volume(occupied, transform, origin, step)
    return torch.minimum(received.clamp(0, 1), coverage)

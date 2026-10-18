"""Isolated learned-content prototype; not yet a detector or an AP result.

The sender learns a pointwise channel basis, with no bias to activate empty
voxels. A complete message has counts 0..3. Prefix k carries min(count, k),
so three binary event layers add to exactly the same full message. ANN control
uses sender-frame channel scales, fixed across all prefixes. Geometry and
session identity are the caller's responsibility; never sum different bases.
"""
import hashlib
import json
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class CountContent:
    counts: torch.Tensor
    scales: torch.Tensor

    def prefix(self, index):
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError('Prefix index must be an integer in 0..3')
        return self.counts.clamp(max=index) * self.scales

    def event(self, index):
        if type(index) is not int or not 1 <= index <= 3:
            raise ValueError('Event index must be an integer in 1..3')
        # Return wire counts, before applying the fixed frame channel scales.
        return self.counts.clamp(max=index) - self.counts.clamp(max=index - 1)


class CountContentBottleneck(nn.Module):
    """128 -> 32 latent channels by default; Count3 and 2-bit ANN controls.

    Group-mean initialization is a transparent lossy starting point, not a
    pretrained semantic code. Trainable projection and expansion have identical
    shapes across families. This code introduces no spatial selection policy.
    """
    def __init__(self, input_channels=128, latent_channels=32, family='count3'):
        super().__init__()
        if (type(input_channels) is not int or type(latent_channels) is not int or
                latent_channels < 1 or input_channels % latent_channels):
            raise ValueError('Positive latent width must divide input width')
        if input_channels < latent_channels or family not in ('count3', 'ann2bit'):
            raise ValueError('Invalid bottleneck width or family')
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.family = family
        self.project = nn.Conv3d(input_channels, latent_channels, 1, bias=False)
        self.expand = nn.Conv3d(latent_channels, input_channels, 1, bias=False)
        group = input_channels // latent_channels
        # Count4's initial 0..4 range maps to the new 0..3 latent range.
        factor = .75 if family == 'count3' else 1.
        with torch.no_grad():
            self.project.weight.zero_()
            self.expand.weight.zero_()
            for channel in range(input_channels):
                latent = channel // group
                self.project.weight[latent, channel, 0, 0, 0] = factor / group
                self.expand.weight[channel, latent, 0, 0, 0] = 1. / factor

    def encode(self, features):
        if (features.ndim != 5 or features.shape[0] != 1 or
                features.shape[1] != self.input_channels or
                any(n < 1 for n in features.shape[2:]) or
                not torch.isfinite(features).all() or torch.any(features < 0)):
            raise ValueError('Expected one finite nonnegative sender frame')
        projected = torch.relu(self.project(features))
        if self.family == 'count3':
            scales = projected.new_ones((1, self.latent_channels, 1, 1, 1))
        else:
            maxima = projected.detach().flatten(2).max(2)[0].reshape(
                1, self.latent_channels, 1, 1, 1)
            # All-zero channels use 1; nonzero channels use sender-only maxima.
            scales = torch.where(maxima > 0, maxima / 3., torch.ones_like(maxima))
            scales = scales.clamp(min=torch.finfo(projected.dtype).tiny)
        continuous = (projected / scales).clamp(0, 3)
        rounded = torch.floor(continuous + .5)
        counts = continuous + (rounded - continuous).detach()
        return CountContent(counts, scales)

    def decode(self, latent):
        # Call after 3D alignment, before HeightCompression. A bias-free linear
        # expansion commutes with trilinear alignment (up to floating error).
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels or not torch.isfinite(latent).all():
            raise ValueError('Invalid aligned latent volume')
        return self.expand(latent)

    def basis_digest(self):
        """Bind both learned matrices, width and quantizer; no stale hash cache."""
        digest = hashlib.sha256(json.dumps(dict(schema=1, family=self.family,
            input_channels=self.input_channels, latent_channels=self.latent_channels),
            sort_keys=True).encode('ascii'))
        for key, value in sorted(self.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode('ascii'))
            digest.update(str(array.dtype).encode('ascii'))
            digest.update(array.tobytes())
        return digest.hexdigest()

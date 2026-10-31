"""Receiver-only spatial reconstruction and multiscale membrane interaction.

No untransmitted roadside features enter this module. The residual is a current
added before Count4, not an ANN path around the spiking readout. A zero final
projection preserves the original F0 function at initialization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout, stride=1):
    # No batch/spatial normalization: a zero remote input remains zero and
    # normalization statistics cannot create global support dependencies.
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.ReLU())


def support_block(mask, stride=1):
    return F.max_pool2d(F.max_pool2d(mask, 3, stride, 1), 3, 1, 1)


class PyramidCurrent(nn.Module):
    def __init__(self, depth, channels=128):
        super().__init__()
        if depth < 1 or channels != 128:
            raise ValueError('Expected positive height and the F0 128-channel basis')
        self.depth, self.channels = depth, channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260929)
            self.spatial = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, bias=False), nn.ReLU(),
                nn.Conv3d(channels, channels, 3, padding=1, bias=False))
            self.local_projection = nn.Conv2d(channels*depth, 128, 1, bias=False)
            self.remote_projection = nn.Conv2d(channels*depth, 128, 1, bias=False)
            self.blocks = nn.ModuleList([block(128, 128), block(128, 256, 2), block(256, 256, 2)])
            self.lateral = nn.ModuleList([nn.Conv2d(c, 128, 1, bias=False) for c in (128,256,256)])
            self.merge = nn.Sequential(nn.Conv2d(384, 256, 3, padding=1, bias=False), nn.ReLU())
            self.output = nn.Conv2d(256, channels*depth, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, membrane, decoded, base_current, received, coverage):
        if membrane.shape != decoded.shape or membrane.shape[1:3] != (self.channels, self.depth):
            raise ValueError('Wrong membrane/decoded shape')
        # Coverage is geometry; received denotes nonzero aligned packet content.
        # Empty messages retain an exactly zero residual even after learning.
        active = received.to(decoded.dtype)
        remote = decoded * active
        remote = remote + self.spatial(remote)
        spatial_support = F.max_pool3d(F.max_pool3d(active, 3, 1, 1), 3, 1, 1)
        remote = remote * spatial_support * (coverage > 0).to(decoded.dtype)
        local = self.local_projection(membrane.flatten(1, 2))
        remote = self.remote_projection(remote.flatten(1, 2))
        mask = spatial_support.amax(2)
        size = local.shape[-2:]
        fused, supports = [], []
        for level, (stage, lateral) in enumerate(zip(self.blocks, self.lateral)):
            local, remote = stage(local), stage(remote)
            mask = support_block(mask, 1 if level == 0 else 2)
            remote = remote * mask
            # Ego query, two source keys/values, separately at each location.
            scores = torch.stack(((local*local).sum(1), (local*remote).sum(1)), 1) / math.sqrt(local.shape[1])
            scores[:, 1:2] = scores[:, 1:2].masked_fill(mask == 0, -1e4)
            weights = scores.softmax(1)
            value = (weights[:, :1]*local + weights[:, 1:]*remote) * mask
            fused.append(F.interpolate(lateral(value), size=size, mode='bilinear', align_corners=False))
            supports.append(F.interpolate(mask, size=size, mode='bilinear', align_corners=False) > 0)
        mask = torch.stack(supports).any(0).to(decoded.dtype)
        mask = F.max_pool2d(mask, 3, 1, 1) > 0
        # Height-compressed interaction can affect any in-range height at the
        # supported XY position; this is a declared receiver reconstruction.
        expanded = mask.unsqueeze(2).expand_as(received) & (coverage > 0)
        delta = self.output(self.merge(torch.cat(fused, 1))).reshape_as(membrane)
        delta = delta * expanded.to(delta.dtype)
        current = base_current * active + delta
        return current, received | expanded

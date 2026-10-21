"""Separate fresh shallow evidence from cacheable deep readout contributions.

The split is before any spatial convolution, BN or activation in shrink.
Only the bias-free deep projection is eligible for a future temporal cache.
This module alone does not implement temporal reuse or regional execution.
"""
import copy
import torch
from torch import nn


class SplitCompactReadout(nn.Module):
    def __init__(self, shrink, current_channels):
        super().__init__()
        if len(shrink.layers) != 1:
            raise ValueError('One compact DoubleConv required')
        sequence = shrink.layers[0].double_conv
        if not isinstance(sequence[0], nn.Sequential) or len(sequence[0]) != 2:
            raise ValueError('Expected projection followed by spatial convolution')
        projection, spatial = sequence[0]
        if not isinstance(projection, nn.Conv2d) or projection.kernel_size != (1,1):
            raise ValueError('Expected a 1x1 projection')
        if projection.bias is not None or projection.groups != 1 or projection.stride != (1,1):
            raise ValueError('Only bias-free ordinary projections can be split')
        if projection.padding != (0,0) or not 0 < current_channels < projection.in_channels:
            raise ValueError('Invalid channel split or projection padding')
        self.current_channels = current_channels
        self.current = nn.Conv2d(current_channels, projection.out_channels, 1, bias=False)
        self.history = nn.Conv2d(projection.in_channels-current_channels, projection.out_channels, 1, bias=False)
        self.current.to(device=projection.weight.device, dtype=projection.weight.dtype)
        self.history.to(device=projection.weight.device, dtype=projection.weight.dtype)
        with torch.no_grad():
            self.current.weight.copy_(projection.weight[:,:current_channels])
            self.history.weight.copy_(projection.weight[:,current_channels:])
        self.spatial = copy.deepcopy(spatial)
        self.tail = copy.deepcopy(nn.Sequential(*list(sequence.children())[1:]))
        self.train(shrink.training)

    def fuse(self, current_projection, historical_projection):
        # Current evidence remains dense; neither BN nor spatial readout is cached.
        return self.tail(self.spatial(current_projection+historical_projection))

    def forward(self, decoded):
        return self.fuse(self.current(decoded[:,:self.current_channels]),
                         self.history(decoded[:,self.current_channels:]))

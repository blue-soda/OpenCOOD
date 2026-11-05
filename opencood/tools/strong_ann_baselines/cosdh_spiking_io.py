"""Spiking sender/receiver replacement for the converted CoSDH path (v2a).

Sender: the ANN encoder (Conv3x3 + BN + ReLU) becomes a single 1x1 spiking
projection; its output is what the IF8 count quantizer transports.
Receiver: the ANN decoder (Conv3x3+BN+ReLU twice) becomes a single 1x1 synapse
followed by a bounded Count16 readout, so the first operation applied to the
received counts is a synapse and a threshold, not a convolution stack.

Both replacements take the centre tap of the convolution they replace, so the
converted model starts close to the source function while changing the
architecture at the communication boundary.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn


class SpikingEncoder(nn.Module):
    def __init__(self, conv):
        super().__init__()
        k = conv.weight.shape[2] // 2
        self.synapse = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
        with torch.no_grad():
            self.synapse.weight.copy_(conv.weight[:, :, k, k].unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        return self.synapse(x)


class SpikingDecoder(nn.Module):
    def __init__(self, conv, theta, steps=16):
        super().__init__()
        k = conv.weight.shape[2] // 2
        self.synapse = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
        with torch.no_grad():
            self.synapse.weight.copy_(conv.weight[:, :, k, k].unsqueeze(-1).unsqueeze(-1))
        self.steps = int(steps)
        self.register_buffer('theta', theta.reshape(1, -1, 1, 1).clone())

    def forward(self, x):
        current = self.synapse(x)
        q = torch.floor(current / self.theta).clamp(0, self.steps) * self.theta
        return current + (q - current).detach()


def install(model, readout_thresholds, steps=16):
    values = json.loads(Path(readout_thresholds).read_text())
    names = []
    for stage, compressor in enumerate(model.naive_compressor_list):
        theta = torch.tensor(values['stages'][str(stage)]['range_p999'],
                             device=next(model.parameters()).device).reshape(-1) / float(steps)
        device = compressor.encoder[0].weight.device
        compressor.encoder = SpikingEncoder(compressor.encoder[0]).to(device)
        compressor.decoder = SpikingDecoder(compressor.decoder[0], theta, steps).to(device)
        names.append('naive_compressor_list.%d.encoder.synapse.weight' % stage)
        names.append('naive_compressor_list.%d.decoder.synapse.weight' % stage)
    return names

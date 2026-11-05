"""Spiking sender/receiver replacement for the converted CoSDH path (v2a).

Sender: the ANN encoder (Conv3x3 + BN + ReLU) becomes a single 1x1 spiking
projection; its output is what the IF8 count quantizer transports.
Receiver: the ANN decoder (Conv3x3+BN+ReLU twice) becomes a single 1x1 synapse
followed by a bounded Count16 readout, so the first operation applied to the
received counts is a synapse and a threshold, not a convolution stack.

Initialisation folds the replaced 3x3 convolutions and their batch norms into
1x1 weights (sum over spatial taps, then BN scale/shift), so the converted
model starts with the same DC gain instead of a degenerate or saturated drive.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn


def _bn_scale(bn):
    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


def _fold(conv, bn):
    with torch.no_grad():
        scale = _bn_scale(bn)
        weight = conv.weight.sum(dim=(2, 3)) * scale[:, None]
        # Keep the rate coder in its responsive range: a folded BN bias can be
        # strongly negative and would silence a whole stage at initialisation.
        bias = bn.bias - bn.running_mean * scale + 1.0
    return weight, bias


class SpikingEncoder(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        weight, bias = _fold(conv, bn)
        self.synapse = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=True)
        with torch.no_grad():
            self.synapse.weight.copy_(weight[:, :, None, None])
            self.synapse.bias.copy_(bias)

    def forward(self, x):
        return torch.relu(self.synapse(x))


class SpikingDecoder(nn.Module):
    def __init__(self, decoder, theta, steps=16):
        super().__init__()
        c0, b0, c1, b1 = decoder[0], decoder[1], decoder[3], decoder[4]
        w0, d0 = _fold(c0, b0)
        w1, d1 = _fold(c1, b1)
        with torch.no_grad():
            weight = w1 @ w0
            bias = w1 @ d0 + d1 + 1.0
        self.synapse = nn.Conv2d(c0.in_channels, c1.out_channels, 1, bias=True)
        with torch.no_grad():
            self.synapse.weight.copy_(weight[:, :, None, None])
            self.synapse.bias.copy_(bias)
        self.steps = int(steps)
        self.register_buffer('theta', theta.reshape(1, -1, 1, 1).clone())

    def forward(self, x):
        drive = torch.relu(self.synapse(x))
        q = torch.floor(drive / self.theta).clamp(0, self.steps) * self.theta
        return drive + (q - drive).detach()


def install(model, readout_thresholds, steps=16):
    values = json.loads(Path(readout_thresholds).read_text())
    names = []
    for stage, compressor in enumerate(model.naive_compressor_list):
        theta = torch.tensor(values['stages'][str(stage)]['range_p999'],
                             device=next(model.parameters()).device).reshape(-1) / float(steps)
        device = compressor.encoder[0].weight.device
        compressor.encoder = SpikingEncoder(compressor.encoder[0], compressor.encoder[1]).to(device)
        compressor.decoder = SpikingDecoder(compressor.decoder, theta, steps).to(device)
        names.append('naive_compressor_list.%d.encoder.synapse.weight' % stage)
        names.append('naive_compressor_list.%d.encoder.synapse.bias' % stage)
        names.append('naive_compressor_list.%d.decoder.synapse.weight' % stage)
        names.append('naive_compressor_list.%d.decoder.synapse.bias' % stage)
    return names

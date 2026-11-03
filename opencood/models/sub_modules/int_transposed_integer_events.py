"""Exact integer event ConvTranspose2d for nonoverlapping upsampling only.

No bias/nonlinearity or quantization is performed here. The original transpose
weight layout is retained; each input event fans out to its own spatial tile.
This is an execution primitive, not an end-to-end model or speed claim.
"""
import torch
from torch import nn


def transposed_integer_events(conv, inputs, weight, chunk=32768):
    if torch.is_grad_enabled():
        raise ValueError('Frozen inference only')
    if (not isinstance(conv, nn.ConvTranspose2d) or conv.groups != 1
            or conv.kernel_size[0] != conv.kernel_size[1]
            or conv.kernel_size not in ((1, 1), (2, 2), (4, 4))
            or conv.stride != conv.kernel_size or conv.padding != (0, 0)
            or conv.output_padding != (0, 0) or conv.dilation != (1, 1)):
        raise ValueError('Nonoverlapping square stride=kernel1/2/4 required')
    if (inputs.dtype != torch.int64 or weight.dtype != torch.int16
            or inputs.ndim != 4 or inputs.shape[0] != 1
            or inputs.shape[1] != conv.in_channels or min(inputs.shape[-2:]) < 1
            or weight.shape != conv.weight.shape or weight.device != inputs.device
            or type(chunk) is not int or chunk < 1):
        raise ValueError('Invalid integer tensors, geometry, device or chunk')
    # Guard before abs(), including the int64 minimum-value overflow case.
    if bool((inputs <= -2**31).any()) or bool((inputs >= 2**31).any()):
        raise ValueError('Input event range exceeded')
    kernel = conv.kernel_size[0]
    mapped = weight.permute(1, 2, 3, 0).reshape(conv.out_channels * kernel**2, conv.in_channels)
    bound = int(mapped.to(torch.int64).abs().sum(1).max()) * int(inputs.abs().max())
    if bound >= 2**53:
        raise ValueError('Exact-reference accumulation bound exceeded')
    _, _, height, width = inputs.shape
    coords = torch.nonzero(inputs[0] != 0, as_tuple=False)
    out = inputs.new_zeros(conv.out_channels * kernel**2, height * width)
    for start in range(0, len(coords), chunk):
        selected = coords[start:start + chunk]
        values = inputs[0, selected[:, 0], selected[:, 1], selected[:, 2]]
        contribution = mapped[:, selected[:, 0]] * values.unsqueeze(0)
        out.index_add_(1, selected[:, 1] * width + selected[:, 2], contribution)
    output = out.reshape(1, conv.out_channels, kernel, kernel, height, width)
    output = output.permute(0, 1, 4, 2, 5, 3).reshape(1, conv.out_channels, height * kernel, width * kernel)
    fanout = conv.out_channels * kernel**2
    return output, dict(nonzero_events=len(coords), unit_spikes=int(inputs.abs().sum()),
        executed_macs=len(coords) * fanout, unit_spike_ac=int(inputs.abs().sum()) * fanout,
        dense_macs=height * width * conv.in_channels * fanout,
        absolute_accumulation_bound=bound, dense_conv_calls=0, chunk_events=chunk)

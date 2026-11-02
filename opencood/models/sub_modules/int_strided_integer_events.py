"""Exact selected-input integer Conv for fine residual-block experiments.

No automatic detector replacement: this is an inference-only arithmetic kernel.
"""
import torch
from torch import nn


def strided_integer_events(conv, delta, weight, chunk=32768):
    if torch.is_grad_enabled():
        raise ValueError('Inference only')
    if not isinstance(conv, nn.Conv2d) or conv.groups != 1 or conv.dilation != (1, 1) or conv.stride not in ((1, 1), (2, 2)) or conv.kernel_size not in ((1, 1), (3, 3)) or conv.padding != (conv.kernel_size[0] // 2,) * 2 or conv.padding_mode != 'zeros':
        raise ValueError('Ungrouped stride1/2 kernel1/3 zero-padded Conv required')
    if delta.ndim != 4 or delta.shape[0] != 1 or delta.shape[1] != conv.in_channels or delta.dtype != torch.int64 or weight.dtype != torch.int16 or weight.shape != conv.weight.shape or weight.device != delta.device or type(chunk) is not int or chunk <= 0:
        raise ValueError('Invalid integer input, weight or chunk')
    if delta.numel() == 0 or bool(((delta <= -(2**32)) | (delta >= 2**32)).any()):
        raise ValueError('Event range exceeded')
    bound = int(weight.to(torch.int64).abs().flatten(1).sum(1).max()) * int(delta.abs().max())
    if bound >= 2**53:
        raise ValueError('Exact accumulation bound exceeded')
    _, _, height, width = delta.shape
    kernel = conv.kernel_size[0]
    pad = kernel // 2
    stride = conv.stride[0]
    oh = (height + 2 * pad - kernel) // stride + 1
    ow = (width + 2 * pad - kernel) // stride + 1
    coords = torch.nonzero(delta[0] != 0, as_tuple=False)
    output = delta.new_zeros(conv.out_channels, oh * ow)
    deliveries = 0
    for ky in range(kernel):
        for kx in range(kernel):
            y = coords[:, 1] + pad - ky
            x = coords[:, 2] + pad - kx
            keep = (y >= 0) & (y < oh * stride) & (x >= 0) & (x < ow * stride) & (y.remainder(stride) == 0) & (x.remainder(stride) == 0)
            selected = coords[keep]
            target = y[keep].div(stride, rounding_mode='floor') * ow + x[keep].div(stride, rounding_mode='floor')
            deliveries += len(selected)
            for start in range(0, len(selected), chunk):
                block = selected[start:start + chunk]
                values = delta[0, block[:, 0], block[:, 1], block[:, 2]]
                contribution = weight[:, block[:, 0], ky, kx] * values.unsqueeze(0)
                output.index_add_(1, target[start:start + chunk], contribution)
    return output.reshape(1, conv.out_channels, oh, ow), dict(
        active_channel_elements=len(coords), scalar_deliveries=deliveries,
        executed_macs=deliveries * conv.out_channels,
        dense_macs=oh * ow * conv.in_channels * conv.out_channels * kernel * kernel,
        absolute_accumulation_bound=bound, chunk_events=chunk)

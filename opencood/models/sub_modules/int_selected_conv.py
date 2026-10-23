"""Selected-output 3x3 convolution for inference experiments.

Gather only neighborhoods whose output cells are requested; do not materialize
an entire-image unfold or compute duplicate halo outputs. This primitive does
not select regions, align caches, or implement recurrent cache ownership.
"""
import torch
from torch import nn
from torch.nn import functional as F


def selected_conv3x3_rows(conv, current, mask, chunk_size=4096):
    """Return selected output rows, flattened locations and executed MACs.

    Input is unpadded and gathers use zero padding even for an externally
    padded Conv2d. No full-frame unfold or unrequested output is computed.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError('Expected Conv2d')
    if (conv.kernel_size != (3, 3) or conv.stride != (1, 1) or
            conv.dilation != (1, 1) or conv.groups != 1 or
            conv.padding not in ((0, 0), (1, 1)) or conv.padding_mode != 'zeros'):
        raise ValueError('Expected zero-padded unit-stride ungrouped 3x3 convolution')
    if conv.training or torch.is_grad_enabled():
        raise ValueError('Frozen inference with gradients disabled is required')
    if current.ndim != 4 or current.shape[0] != 1 or current.shape[1] != conv.in_channels:
        raise ValueError('Expected one unpadded input frame')
    height, width = current.shape[-2:]
    if height < 1 or width < 1 or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Invalid grid or chunk size')
    if mask.shape != (1, 1, height, width) or mask.dtype != torch.bool:
        raise ValueError('Expected matching boolean output mask')
    if (mask.device != current.device or
            conv.weight.device != current.device or
            conv.weight.dtype != current.dtype):
        raise ValueError('Input, mask and weights must share device and floating dtype')
    selected = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    count = selected.numel()
    if not count:
        return current.new_empty((0, conv.out_channels)), selected, 0
    rows = current[0].permute(1, 2, 0).reshape(height * width, conv.in_channels)
    rows = torch.cat((rows, rows.new_zeros((1, conv.in_channels))), dim=0)
    result = current.new_empty((count, conv.out_channels))
    weight = conv.weight.permute(0, 2, 3, 1).reshape(conv.out_channels, -1)
    offsets = torch.arange(-1, 2, device=current.device)
    for start in range(0, count, chunk_size):
        target = selected[start:start + chunk_size]
        yy = torch.div(target[:, None, None], width, rounding_mode='floor') + offsets[None, :, None]
        xx = target[:, None, None] % width + offsets[None, None, :]
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        indices = yy * width + xx
        indices = torch.where(valid, indices, indices.new_full((), height * width))
        patches = rows[indices.reshape(-1)].reshape(target.numel(), -1)
        result[start:start + target.numel()] = F.linear(patches, weight, conv.bias)
    return result, selected, count * conv.in_channels * conv.out_channels * 9


def selected_conv3x3(conv, current, aligned_cache, mask, chunk_size=4096):
    """Scatter selected pre-BN responses into the supplied aligned cache.

    MACs omit indexing/copying/scatter; those costs remain in wall time.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError('Expected Conv2d')
    if aligned_cache.shape != (1, conv.out_channels, *current.shape[-2:]):
        raise ValueError('Expected matching aligned output cache')
    if aligned_cache.device != current.device or aligned_cache.dtype != current.dtype:
        raise ValueError('Input and cache must share device and dtype')
    rows, locations, macs = selected_conv3x3_rows(conv, current, mask, chunk_size)
    output = aligned_cache.reshape(1, conv.out_channels, -1).clone()
    output[0, :, locations] = rows.t()
    return output.reshape_as(aligned_cache), macs

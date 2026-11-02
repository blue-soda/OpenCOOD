"""Per-input spatial bases followed by pointwise mixing; no temporal cache."""
import copy
import torch
from torch import nn


def spatial_basis_conv(conv, rank):
    if not isinstance(conv, nn.Conv2d) or conv.groups != 1 or conv.padding_mode != 'zeros':
        raise ValueError('Ungrouped zero-padded Conv2d required')
    maximum = min(conv.out_channels, conv.kernel_size[0] * conv.kernel_size[1])
    if type(rank) is not int or not 1 <= rank <= maximum:
        raise ValueError('Invalid spatial rank')
    weight = conv.weight.detach()
    matrix = weight.cpu().double().permute(1, 0, 2, 3).reshape(conv.in_channels, conv.out_channels, -1)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    spatial = nn.Conv2d(conv.in_channels, conv.in_channels * rank, conv.kernel_size,
                        conv.stride, conv.padding, conv.dilation, groups=conv.in_channels,
                        bias=False).to(weight.device, weight.dtype)
    pointwise = nn.Conv2d(conv.in_channels * rank, conv.out_channels, 1,
                         bias=conv.bias is not None).to(weight.device, weight.dtype)
    with torch.no_grad():
        spatial.weight.copy_(vh[:, :rank].reshape_as(spatial.weight))
        pointwise.weight.copy_((u[:, :, :rank] * s[:, None, :rank]).permute(1, 0, 2).reshape_as(pointwise.weight))
        if conv.bias is not None:
            pointwise.bias.copy_(conv.bias)
    return nn.Sequential(spatial, pointwise)


def spatial_basis_readout(source, rank=4):
    result = copy.deepcopy(source)
    changed = []
    def visit(module, prefix=''):
        for name, child in list(module.named_children()):
            path = prefix + name
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                r = min(child.out_channels, 9) if rank is None else rank
                setattr(module, name, spatial_basis_conv(child, r))
                changed.append(dict(path=path, input=child.in_channels, output=child.out_channels, rank=r))
            else:
                visit(child, path + '.')
    visit(result)
    if not changed:
        raise ValueError('No spatial readout convolutions found')
    return result, changed

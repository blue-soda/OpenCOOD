"""High-resolution head with a wider ANN BEV backbone and unchanged head width.

The base snapshot model keeps a small ANN BEV (its `BaseBEVBackbone_spike`
modules have been switched to ReLU by the snapshot adapter), so widening must
preserve that exact module family. This class deep-copies the constructed
backbone and rewrites every convolution and batch-norm in place: 64->96 and
128->160 channels, while the deblock output channels stay (128, 128) and the
256-channel head input, local-detail sum and all downstream shapes are
unchanged. The preparation script applies the same function to a backbone
loaded from the source checkpoint, so training starts from an exact warm start.
"""
import copy

import torch
import torch.nn as nn

from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres

CHANNELS = {64: 96, 128: 160}


def _widen_channels(value):
    return CHANNELS.get(value, value)


def _expand_conv(old, new):
    with torch.no_grad():
        weight = torch.zeros_like(new.weight)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(old.weight.shape, new.weight.shape))
        weight[sl] = old.weight[sl]
        new.weight.copy_(weight)


def _expand_bn(old, new):
    n = old.num_features
    with torch.no_grad():
        new.weight.zero_(); new.bias.zero_()
        new.running_mean.zero_(); new.running_var.fill_(1.)
        new.weight[:n].copy_(old.weight)
        new.bias[:n].copy_(old.bias)
        new.running_mean[:n].copy_(old.running_mean)
        new.running_var[:n].copy_(old.running_var)


def _widen(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.ConvTranspose2d):
            new = type(child)(_widen_channels(child.in_channels),
                              _widen_channels(child.out_channels),
                              child.kernel_size, child.stride, child.padding,
                              child.dilation, child.groups,
                              child.bias is not None, child.padding_mode)
            _expand_conv(child, new)
            module[int(name) if isinstance(module, nn.Sequential) else name] = new
        elif isinstance(child, nn.BatchNorm2d):
            new = nn.BatchNorm2d(_widen_channels(child.num_features),
                                 eps=child.eps, momentum=child.momentum,
                                 affine=child.affine, track_running_stats=child.track_running_stats)
            _expand_bn(child, new)
            module[int(name) if isinstance(module, nn.Sequential) else name] = new
        else:
            _widen(child)


def widen_bev(backbone):
    widened = copy.deepcopy(backbone)
    _widen(widened)
    return widened


class E3dsnnSnapshotHighresWide(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        upsample = tuple(int(x) for x in args.get('bev_wide', {}).get('upsample_filters', (128, 128)))
        if upsample != (128, 128):
            raise ValueError('Head input must stay 256: upsample filters are fixed')
        self.backbone_2d = widen_bev(self.backbone_2d)

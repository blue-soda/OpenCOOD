"""High-resolution head with a wider ANN BEV backbone and unchanged head width.

The base snapshot model keeps a small ANN BEV (the adapter switches the
spiking BEV modules to ReLU), so widening must preserve that exact module
family. This class widens the already-constructed backbone in place: every
convolution and batch-norm maps 64->96 and 128->160 channels, while the deblock
output channels stay (128, 128) and the 256-channel head input, local-detail sum
and all downstream shapes are unchanged. The preparation script applies the
same function to a backbone loaded from the source checkpoint, so training
starts from an exact warm start.
"""
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


def widen_inplace(module, widen_out=True):
    """Rewrite every Conv2d/ConvTranspose2d/BatchNorm2d inside ``module``.

    ``widen_out=False`` keeps output channels unchanged; used for the deblocks,
    whose 128-channel outputs feed the fixed 256-channel head input.
    """
    for name, child in list(module.named_children()):
        target = int(name) if isinstance(module, nn.Sequential) else name
        grow = widen_out and name != 'deblocks'
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            out = _widen_channels(child.out_channels) if grow else child.out_channels
            if isinstance(child, nn.ConvTranspose2d):
                new = nn.ConvTranspose2d(_widen_channels(child.in_channels), out,
                                         child.kernel_size, child.stride, child.padding,
                                         output_padding=child.output_padding,
                                         groups=child.groups, bias=child.bias is not None,
                                         dilation=child.dilation,
                                         padding_mode=child.padding_mode)
            else:
                new = nn.Conv2d(_widen_channels(child.in_channels), out,
                                child.kernel_size, child.stride, child.padding,
                                child.dilation, child.groups,
                                child.bias is not None, child.padding_mode)
            _expand_conv(child, new)
            if isinstance(module, nn.Sequential):
                module[target] = new
            else:
                setattr(module, target, new)
        elif isinstance(child, nn.BatchNorm2d):
            feats = _widen_channels(child.num_features) if grow else child.num_features
            new = nn.BatchNorm2d(feats, eps=child.eps, momentum=child.momentum,
                                 affine=child.affine,
                                 track_running_stats=child.track_running_stats)
            _expand_bn(child, new)
            if isinstance(module, nn.Sequential):
                module[target] = new
            else:
                setattr(module, target, new)
        else:
            widen_inplace(child, grow)


class E3dsnnSnapshotHighresWide(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        upsample = tuple(int(x) for x in args.get('bev_wide', {}).get('upsample_filters', (128, 128)))
        if upsample != (128, 128):
            raise ValueError('Head input must stay 256: upsample filters are fixed')
        widen_inplace(self.backbone_2d)

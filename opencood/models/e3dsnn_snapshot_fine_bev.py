"""Finer BEV input: keep x/y at stride 4 while preserving the channel count.

The stock encoder downsamples x/y by 2 in `conv4`, so the ego dense feature is
1.6 m. Replacing that one sparse convolution with stride (2, 1, 1) keeps the z
downsample (and therefore 128*z channels, so every weight shape is unchanged)
while x/y stay at 0.8 m. The 2D backbone then sees twice the linear resolution,
and the head upsamples once more (factor 2) to 0.4 m.

Weights are inherited exactly; only the sparse layer's stride and indice key
change, so this is a resolution intervention, not a capacity one.
"""
import spconv.pytorch as spconv
import torch

from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres

FINE_STRIDE = (2, 1, 1)


def restore_fine_conv4(module, stride=FINE_STRIDE, indice_key='spconv4fine'):
    """Swap conv4's downsample conv for a stride-(2,1,1) copy of its weights."""
    down = module.down
    if not isinstance(down, spconv.SparseSequential) or len(down) < 3:
        raise ValueError('Unexpected conv4 downsampling layout')
    children = list(down)
    old = children[1]
    if not isinstance(old, spconv.SparseConv3d):
        raise ValueError('Unexpected conv4 downsampling layer')
    replacement = spconv.SparseConv3d(old.in_channels, old.out_channels,
                                      kernel_size=old.kernel_size, stride=stride,
                                      padding=old.padding, indice_key=indice_key,
                                      bias=False)
    with torch.no_grad():
        if tuple(replacement.weight.shape) != tuple(old.weight.shape):
            raise ValueError('Fine-BEV conv keeps the weight shape')
        replacement.weight.copy_(old.weight)
    children[1] = replacement
    module.down = spconv.SparseSequential(*children)
    return module


class E3dsnnSnapshotFineBev(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        if self.upsample_factor != 2:
            raise ValueError('Fine-BEV head requires the factor-2 upsample')
        restore_fine_conv4(self.backbone_3d.conv4)

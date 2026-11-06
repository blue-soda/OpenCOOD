"""Finer BEV input: keep x/y at stride 4 while preserving the channel count.

The stock encoder downsamples x/y by 2 in `conv4`, so the ego dense feature is
1.6 m. Replacing that one sparse convolution with stride (2, 1, 1) keeps the z
downsample (and therefore 128*z channels, so every weight shape is unchanged)
while x/y stay at 0.8 m. The 2D backbone then sees twice the linear resolution,
and the head upsamples once more (factor 2) to 0.4 m.

The physical grid contract must change with the stride: `encoded_grid` hard-codes
the 8x x/y step and the 8.5-voxel z origin of the stock chain. Leaving it
unchanged makes the message contract declare 1.6 m cells for an 0.8 m grid, so
the receiver-side warp is wrong whenever the sender pose is not identity. This
module overrides the contract with the values implied by the modified chain:
x/y step 4 voxels (center preserved by the k=3, s=1, p=1 layer), z step and
origin unchanged.

Weights are inherited exactly; only the sparse layer's stride and the geometry
contract change, so this is a resolution intervention, not a capacity one.
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


def fine_grid(voxel_size, lidar_range):
    """Physical origin/step of the fine chain (xyz, metres).

    x/y keep the k=3, s=1, p=1 center, so the 0.5-voxel origin offset is
    unchanged while the step becomes 4 voxels. z still passes the stock
    s=2 chain and the final k=(3,1,1), s=(2,1,1), p=0 layer, so its step is
    16 voxels and its origin offset 8.5 voxels.
    """
    origin = (lidar_range[0] + 0.5 * voxel_size[0],
              lidar_range[1] + 0.5 * voxel_size[1],
              lidar_range[2] + 8.5 * voxel_size[2])
    step = (4 * voxel_size[0], 4 * voxel_size[1], 16 * voxel_size[2])
    return origin, step


class E3dsnnSnapshotFineBev(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        if self.upsample_factor != 2:
            raise ValueError('Fine-BEV head requires the factor-2 upsample')
        restore_fine_conv4(self.backbone_3d.conv4)
        self.origin, self.step = fine_grid(args['voxel_size'], args['lidar_range'])

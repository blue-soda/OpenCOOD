"""Depth capacity variant with normally initialized residual blocks.

The stock depth variant zeroes the last BatchNorm affine of both branches of
every appended block so that the model is exactly equivalent to its source at
initialization. That equivalence also blocks gradients to the branch
convolutions until the scale moves, so the extra capacity stays inactive under
a small learning rate.

This variant keeps the same architecture and parameter names but initializes
those scales to one, making the appended blocks active from the first step.
Use it to test whether the depth capacity helps once it is actually trained.
"""
import torch

from opencood.models.e3dsnn_snapshot_highres_depth import E3dsnnSnapshotHighresDepth


class E3dsnnSnapshotHighresDepthActive(E3dsnnSnapshotHighresDepth):
    def __init__(self, args):
        super().__init__(args)
        depth = args['frontend']['extra_blocks_per_stage']
        for stage in (3, 4):
            blocks = getattr(self.backbone_3d, 'conv' + str(stage)).blocks
            for block in blocks[-depth:]:
                for branch in (block.block, block.voxel_block):
                    with torch.no_grad():
                        torch.nn.init.ones_(branch[-1].weight)
                        torch.nn.init.zeros_(branch[-1].bias)

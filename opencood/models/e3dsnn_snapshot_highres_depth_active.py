"""Depth capacity variant with normally initialized residual blocks.

The stock depth variant zeroes the last BatchNorm affine of both branches of
every appended block so that the model is exactly equivalent to its source at
initialization. That equivalence also blocks gradients to the branch
convolutions until the scale moves, so the extra capacity stays inactive under
a small learning rate.

This variant keeps the same architecture and parameter names but sets those
scales to one, making the appended blocks active from the first step.

The activation must also survive checkpoint loading: the trainer constructs the
model and then calls ``load_state_dict(strict=True)``, which would restore the
zero scales from a stock depth checkpoint. ``load_state_dict`` is therefore
overridden to re-apply the active initialization after every load, and
``activation_applied`` records that this happened.
"""
import torch

from opencood.models.e3dsnn_snapshot_highres_depth import E3dsnnSnapshotHighresDepth


class E3dsnnSnapshotHighresDepthActive(E3dsnnSnapshotHighresDepth):
    def __init__(self, args):
        super().__init__(args)
        self.extra_depth = int(args['frontend']['extra_blocks_per_stage'])
        if self.extra_depth <= 0:
            raise ValueError('Active depth variant requires extra blocks')
        self.activation_applied = False
        self.activate_appended_blocks()

    def activate_appended_blocks(self):
        for stage in (3, 4):
            blocks = getattr(self.backbone_3d, 'conv' + str(stage)).blocks
            for block in blocks[-self.extra_depth:]:
                for branch in (block.block, block.voxel_block):
                    with torch.no_grad():
                        torch.nn.init.ones_(branch[-1].weight)
                        torch.nn.init.zeros_(branch[-1].bias)
        self.activation_applied = True

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self.activate_appended_blocks()
        return result

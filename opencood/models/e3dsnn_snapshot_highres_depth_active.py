"""Depth capacity variant whose appended residual blocks start active.

The stock depth variant zeroes the last BatchNorm affine of both branches of
every appended block, so the model is exactly equivalent to its source at
initialization. That equivalence also blocks gradients to the branch
convolutions until the scale moves, so the extra capacity can stay inactive
under a small learning rate.

This class only performs a one-time preparation: ``activate_appended_blocks``
sets those scales to one for a *fresh* model. It deliberately does NOT override
``load_state_dict``: a trained checkpoint already carries whatever scales the
optimizer learned, and re-activating on every load would silently reset them.

To create an active starting point, build this model, load the source
checkpoint, call ``activate_appended_blocks()`` and save the resulting state
dict with ``opencood.tools.prepare_active_initial``. Train from that file with
the ordinary training command.
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

    def activate_appended_blocks(self):
        for stage in (3, 4):
            blocks = getattr(self.backbone_3d, 'conv' + str(stage)).blocks
            for block in blocks[-self.extra_depth:]:
                for branch in (block.block, block.voxel_block):
                    with torch.no_grad():
                        torch.nn.init.ones_(branch[-1].weight)
                        torch.nn.init.zeros_(branch[-1].bias)
        self.activation_applied = True
        return self

"""Matched full-encoder adaptation, with optional identity-initialized depth.

Keep the Scheme-2 IF8 message, alignment and small ANN BEV unchanged. Extra
blocks operate on sparse membrane features before conv_out/Count4 and message
projection; this is an encoder capacity experiment, not a new wire mechanism.
"""
from functools import partial

import torch
from torch import nn

from opencood.models.e3dsnn_snapshot_joint import E3dsnnSnapshotJoint, JOINT_PREFIXES
from opencood.models.sub_modules.e3dsnn.backbone_3d import BasicBlock


class E3dsnnSnapshotFrontend(E3dsnnSnapshotJoint):
    def __init__(self, args):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(args)
        depth = args['frontend']['extra_blocks_per_stage']
        if type(depth) is not int or depth not in (0, 2):
            raise ValueError('Registered comparison adds zero or two blocks at stages 3/4')
        if args['receiver_activation'] != 'relu' or self.snapshot_family != 'if8':
            raise ValueError('Retain small ANN BEV and Scheme-2 IF8')
        # Both arms retain identical global RNG state for augmentation/dropout.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260929)
            for stage in (3, 4):
                blocks = getattr(self.backbone_3d, 'conv' + str(stage)).blocks
                for _ in range(depth):
                    block = BasicBlock(64, 64,
                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=.01),
                        indice_key='subm' + str(stage), groups=8 if stage == 3 else 16)
                    # Both residual branches return exact zero at initialization.
                    # Interior BN stays nonzero so the last affine scale can learn.
                    for norm in (block.block[-1], block.voxel_block[-1]):
                        nn.init.zeros_(norm.weight)
                        nn.init.zeros_(norm.bias)
                    blocks.append(block)
        self.freeze_encoder = False
        self.train_encoder = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(('backbone_3d.',) + JOINT_PREFIXES))
        # E3dsnnSnapshotJoint.train keeps every running BN statistic frozen;
        # affine parameters, including new residual branch scales, can train.

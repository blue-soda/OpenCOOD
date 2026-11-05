"""High-resolution head with the matched int8/ANN2bit sender control.

Same stride-4 head, local detail, joint training and initialisation as the IF8
arm; only the message representation changes (raw L1 row selection, 2-bit
per-channel scales) and the receiver is the ANN content accumulator instead of
the membrane readout. This isolates the sender representation for the mechanism
table. The base classes restrict the snapshot family to IF8, so this class
bypasses their constructors while reusing their methods.
"""
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres
from opencood.models.e3dsnn_snapshot_joint import E3dsnnSnapshotJoint, JOINT_PREFIXES
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.sub_modules.e3dsnn_highres_bev import LocalCountDetail


class E3dsnnSnapshotHighresAnn(E3dsnnSnapshotHighres):
    def __init__(self, args):
        E3dsnnSnapshotJoint.__init__(self, args)
        opts = args['highres']
        if opts['output_stride'] != 4:
            raise ValueError('ANN2bit control keeps the stride-4 head')
        if args['frontend']['extra_blocks_per_stage'] != 0:
            raise ValueError('ANN2bit control uses the base backbone depth')
        self.use_shallow_skip = opts['use_shallow_skip']
        if type(self.use_shallow_skip) is not bool:
            raise ValueError('use_shallow_skip must be boolean')
        self.local_detail = LocalCountDetail(
            opts.get('hidden_channels', 128),
            opts.get('second_dilation', 1)) if self.use_shallow_skip else None
        self.shallow_readout = Multispike()
        self._local_shallow = None
        self.freeze_encoder = False
        self.train_encoder = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(('backbone_3d.',) + JOINT_PREFIXES))

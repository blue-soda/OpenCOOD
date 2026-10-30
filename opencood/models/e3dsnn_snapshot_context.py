"""Scheme-2 F0 with an optional message-conditioned supplemental receiver gate.

Two equal-capacity arms differ only in the second gate input. Sender, codec,
geometry, Count4 and ANN detector remain the F0 graph. Not an event-time claim.
"""
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.models.fuse_modules.e3dsnn_context_gain import SnapshotContextGain


class E3dsnnSnapshotContext(E3dsnnSnapshotFrontend):
    def __init__(self, args):
        super().__init__(args)
        if args['frontend']['extra_blocks_per_stage'] != 0:
            raise ValueError('Context comparison uses original F0 depth')
        self.joint_receiver.snapshot_context_gate = SnapshotContextGain(
            args['snapshot_context']['condition'])

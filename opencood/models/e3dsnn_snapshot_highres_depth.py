"""High-resolution head with registered extra depth at stages 3/4.

`E3dsnnSnapshotHighres` deliberately restricts the depth search to the original
backbone. The capacity arm needs the same stride-4 head and local detail with
`frontend.extra_blocks_per_stage == 2`, so this class bypasses the parent
constructor (not its methods) and repeats the shape checks without the depth
restriction. Added blocks are zero-initialised residual branches created by
`E3dsnnSnapshotFrontend`, so the model starts bit-identical to the base one.
"""
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.sub_modules.e3dsnn_highres_bev import LocalCountDetail


class E3dsnnSnapshotHighresDepth(E3dsnnSnapshotHighres):
    def __init__(self, args):
        E3dsnnSnapshotFrontend.__init__(self, args)
        opts = args['highres']
        if opts['output_stride'] != 4:
            raise ValueError('Registered depth arm keeps the stride-4 head')
        if args['frontend']['extra_blocks_per_stage'] not in (0, 2):
            raise ValueError('Registered depth arm adds zero or two blocks')
        self.use_shallow_skip = opts['use_shallow_skip']
        if type(self.use_shallow_skip) is not bool:
            raise ValueError('use_shallow_skip must be boolean')
        self.local_detail = LocalCountDetail(
            opts.get('hidden_channels', 128),
            opts.get('second_dilation', 1)) if self.use_shallow_skip else None
        self.shallow_readout = Multispike()
        self.upsample_factor = 2
        self._local_shallow = None

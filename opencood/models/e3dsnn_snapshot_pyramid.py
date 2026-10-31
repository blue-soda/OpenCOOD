"""Full-front SNN/IF8 with a larger multiscale receiver before Count4."""
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.models.fuse_modules.e3dsnn_pyramid_current import PyramidCurrent


class E3dsnnSnapshotPyramid(E3dsnnSnapshotFrontend):
    def __init__(self, args):
        super().__init__(args)
        if args['frontend']['extra_blocks_per_stage'] != 0:
            raise ValueError('Pyramid receiver starts from original F0 depth')
        extent = args['lidar_range'][5] - args['lidar_range'][2]
        depth = int(round(extent / args['voxel_size'][2])) + 1
        for _ in range(3):
            depth = (depth + 1) // 2
        depth = (depth - 3) // 2 + 1
        self.joint_receiver.snapshot_pyramid = PyramidCurrent(depth)

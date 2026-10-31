"""F0 with identity-initialized, source-only spatial/global IF modulation."""
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.models.fuse_modules.e3dsnn_spatial_drive import SnapshotSpatialDrive


class E3dsnnSnapshotSpatial(E3dsnnSnapshotFrontend):
    def __init__(self, args):
        super().__init__(args)
        if args['frontend']['extra_blocks_per_stage'] != 0:
            raise ValueError('Spatial drive comparison uses original F0 depth')
        self.snapshot_spatial_drive = SnapshotSpatialDrive(args['snapshot_spatial']['condition'])

    def snapshot_drive(self, source):
        drive = super().snapshot_drive(source)
        return self.snapshot_spatial_drive(source, drive)

    def event_basis_digest(self):
        return self.snapshot_spatial_drive.basis_digest(super().event_basis_digest())

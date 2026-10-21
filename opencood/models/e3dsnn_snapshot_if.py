from opencood.models.fuse_modules.e3dsnn_snapshot_adapter import SnapshotAdapter
from opencood.models.e3dsnn_event_detector import E3dsnnEventDetector


class E3dsnnSnapshotIf(SnapshotAdapter, E3dsnnEventDetector):
    pass

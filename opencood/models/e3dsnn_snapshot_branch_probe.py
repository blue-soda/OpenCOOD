"""Representative single-path alternatives with the same shallow IF message.

The removed path is absent from the parameter set, not merely zeroed at the
head. All arms retain the original detection grid and local auxiliary loss.
"""
import spconv.pytorch as spconv
from opencood.models.e3dsnn_snapshot_early_message import E3dsnnSnapshotEarlyMessage


class E3dsnnSnapshotBranchProbe(E3dsnnSnapshotEarlyMessage):
    def __init__(self, args):
        super().__init__(args)
        self.branch_mode = args['branch_mode']
        if self.branch_mode not in ('dual', 'deep', 'shallow'):
            raise ValueError('Expected dual, deep or shallow branch mode')
        if self.branch_mode == 'deep':
            self.local_detail = None
            self.use_shallow_skip = False
        elif self.branch_mode == 'shallow':
            self.backbone_3d.conv4 = None
            self.backbone_3d.conv_out = None
            self.backbone_2d = None

    def encode_membrane(self, lidar, batch_size):
        if self.branch_mode != 'shallow':
            return super().encode_membrane(lidar, batch_size)
        n = lidar['voxel_num_points'].view(-1, 1).to(lidar['voxel_features'].dtype)
        if not len(n):
            raise ValueError('Empty ego voxel batch')
        mean = lidar['voxel_features'].sum(1) / n.clamp_min(1)
        x = spconv.SparseConvTensor(mean, lidar['voxel_coords'].int(),
                                   self.backbone_3d.sparse_shape, int(batch_size))
        for layer in (self.backbone_3d.conv_input, self.backbone_3d.conv1,
                      self.backbone_3d.conv2, self.backbone_3d.conv3):
            x = layer(x)
        membrane = x.dense()
        counts = x.replace_feature(self.shallow_readout(x.features))
        self._shallow_shape = tuple(counts.spatial_shape)
        self._local_shallow = counts.dense().amax(2)
        if self.training:
            self._local_bev = self._local_shallow
        return counts, membrane

    def _detect_shallow(self, counts):
        bev = self.local_detail(counts)
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))

    def detect_bev_sample(self, spatial, sample_index):
        if self.branch_mode != 'shallow':
            return super().detect_bev_sample(spatial, sample_index)
        counts = spatial.reshape(1, 64, *self._shallow_shape)
        return self._detect_shallow(counts.amax(2))

    def detect_bev(self, spatial):
        if self.branch_mode != 'shallow':
            return super().detect_bev(spatial)
        return self._detect_shallow(spatial)

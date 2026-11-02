"""Propagate the single shallow IF message through the existing SNN depth.

The received Count4 field feeds both the original conv4/conv_out and the
existing high-resolution detail module. No new parameters or second payload.
"""
import torch
import spconv.pytorch as spconv
from opencood.models.e3dsnn_snapshot_shallow_message import E3dsnnSnapshotShallowMessage
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike


class E3dsnnSnapshotEarlyMessage(E3dsnnSnapshotShallowMessage):
    def __init__(self, args):
        super().__init__(args)
        if not isinstance(self.backbone_3d.conv4.down[0], Multispike):
            raise ValueError('Expected preactivation Count4 at the conv4 entrance')
        self._ego_indices = None

    def _deep_from_counts(self, counts):
        # The fused input has already passed Count4. Applying it again changes
        # its surrogate at integer 0/4, so skip precisely that activation.
        x = counts
        for layer in list(self.backbone_3d.conv4.down)[1:]:
            x = layer(x)
        for block in self.backbone_3d.conv4.blocks:
            x = block(x)
        return self.backbone_3d.conv_out(x)

    def encode_membrane(self, lidar, batch_size):
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
        self._ego_indices = counts.indices
        self._shallow_shape = tuple(counts.spatial_shape)
        self._local_shallow = counts.dense().amax(2)
        if self.training:
            # Required only by the existing ego auxiliary loss; no duplicate
            # ego deep pass at inference.
            self._local_bev = self._deep_from_counts(counts).dense().flatten(1, 2)
        return counts, membrane

    def _fused_sparse(self, dense, sample_index):
        if self._ego_indices is None or self._shallow_shape is None:
            raise ValueError('Missing frame-local ego support')
        original = self._ego_indices[self._ego_indices[:, 0] == sample_index].clone()
        original[:, 0] = 0
        occupied = torch.zeros(self._shallow_shape, dtype=torch.bool, device=dense.device)
        pos = original[:, 1:].long()
        occupied[pos[:, 0], pos[:, 1], pos[:, 2]] = True
        # Preserve all original ego sites, including zero-count sites. New road
        # sites require an actual nonzero received Count4, not a continuous or
        # merely geometric support bypass. Topology selection is discrete.
        extra = ((dense.detach()[0] != 0).any(0) & ~occupied).nonzero(as_tuple=False)
        extra = torch.cat((torch.zeros_like(extra[:, :1]), extra), dim=1).to(original.dtype)
        indices = torch.cat((original, extra), dim=0)
        pos = indices[:, 1:].long()
        features = dense[0, :, pos[:, 0], pos[:, 1], pos[:, 2]].t().contiguous()
        return spconv.SparseConvTensor(features, indices, self._shallow_shape, 1)

    def detect_bev_sample(self, spatial, sample_index):
        counts = spatial.reshape(1, 64, *self._shallow_shape)
        deep = self._deep_from_counts(self._fused_sparse(counts, sample_index))
        return self._detect_highres(deep.dense().flatten(1, 2), counts.amax(2))

    def forward(self, data):
        try:
            return super().forward(data)
        finally:
            self._ego_indices = None

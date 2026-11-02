"""One paid IF8 message at stride 4, feeding the existing Count4 detail path.

The ego deep path and ANN detector are preserved. Road encoding ends at conv3;
no second payload, confidence network, or additional detail module is introduced.
"""
import torch
import spconv.pytorch as spconv
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres
from opencood.models.fuse_modules.e3dsnn_content_message import CountContentBottleneck
from opencood.models.fuse_modules.e3dsnn_joint_membrane import JointMembraneReceiver


class E3dsnnSnapshotShallowMessage(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        if not self.use_shallow_skip or self.snapshot_family != 'if8':
            raise ValueError('Shallow message requires the preserved Count4 detail and IF8')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261102)
            self.content = CountContentBottleneck(input_channels=64, latent_channels=32)
            self.joint_receiver = JointMembraneReceiver(channels=64, latent_channels=32)
        # Two k=3, stride=2, padding=1 convolutions preserve the voxel center.
        self.origin = tuple(float(x + .5 * v) for x, v in
                            zip(args['lidar_range'][:3], args['voxel_size']))
        self.step = tuple(float(4 * v) for v in args['voxel_size'])
        self._ego_deep = None
        self._shallow_shape = None

    def encode_membrane(self, lidar, batch_size):
        captured = []
        handle = self.backbone_3d.conv3.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            # Explicit base dispatch avoids calling the road-only prefix below.
            deep = E3dsnnSingle.encode_lidar(self, lidar, batch_size)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('Expected exactly one ego shallow encoding')
        shallow = captured[0]
        membrane = shallow.dense()
        counts = shallow.replace_feature(self.shallow_readout(shallow.features))
        self._local_shallow = counts.dense().amax(2)
        self._ego_deep = deep.dense().flatten(1, 2)
        self._shallow_shape = tuple(shallow.spatial_shape)
        if self.training:
            self._local_bev = self._ego_deep
        return counts, membrane

    def encode_lidar(self, lidar, batch_size):
        # The road needs only conv3. Do not compute and discard conv4/conv_out.
        n = lidar['voxel_num_points'].view(-1, 1).to(lidar['voxel_features'].dtype)
        if not len(n):
            raise ValueError('Empty road voxel batch')
        mean = lidar['voxel_features'].sum(1) / n.clamp_min(1)
        x = spconv.SparseConvTensor(mean, lidar['voxel_coords'].int(),
                                   self.backbone_3d.sparse_shape, int(batch_size))
        for layer in (self.backbone_3d.conv_input, self.backbone_3d.conv1,
                      self.backbone_3d.conv2, self.backbone_3d.conv3):
            x = layer(x)
        return x.replace_feature(self.shallow_readout(x.features))

    def detect_bev_sample(self, spatial, sample_index):
        if self._ego_deep is None or self._shallow_shape is None:
            raise ValueError('Missing frame-local ego features')
        counts = spatial.reshape(1, 64, *self._shallow_shape)
        return self._detect_highres(self._ego_deep[sample_index:sample_index+1],
                                    counts.amax(2))

    def forward(self, data):
        try:
            return super().forward(data)
        finally:
            self._ego_deep = None
            self._shallow_shape = None

"""Stride-4 head with optional ego-only, shallow SNN count detail.

The road still follows the paid IF8 -> 3D warp -> membrane -> Count4 path.
The paired interpolation control uses the same head grid and pretrained layers.
"""
from opencood.models.e3dsnn_snapshot_frontend import E3dsnnSnapshotFrontend
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
import torch.nn.functional as F
from opencood.models.sub_modules.e3dsnn_highres_bev import LocalCountDetail, upsample_bev


class E3dsnnSnapshotHighres(E3dsnnSnapshotFrontend):
    def __init__(self, args):
        super().__init__(args)
        opts = args['highres']
        if opts['output_stride'] not in (4, 2) or args['frontend']['extra_blocks_per_stage'] != 0:
            raise ValueError('Registered high-resolution experiment uses original depth and stride 4')
        self.use_shallow_skip = opts['use_shallow_skip']
        if type(self.use_shallow_skip) is not bool:
            raise ValueError('use_shallow_skip must be boolean')
        self.local_detail = LocalCountDetail(opts.get('hidden_channels', 128),
                                            opts.get('second_dilation', 1)) if self.use_shallow_skip else None
        self.upsample_factor = int(opts.get('upsample_factor', 2))
        if self.upsample_factor not in (2, 4):
            raise ValueError('Head upsample factor must be 2 or 4')
        self.shallow_readout = Multispike()
        self._local_shallow = None

    def encode_membrane(self, lidar, batch_size):
        if not self.use_shallow_skip:
            return super().encode_membrane(lidar, batch_size)
        def capture(module, inputs, sparse):
            # Apply Count4 before height reduction; never use a continuous
            # shallow membrane bypass. Hook exists only during ego encoding.
            counts = sparse.replace_feature(self.shallow_readout(sparse.features))
            self._local_shallow = counts.dense().amax(2)
        handle = self.backbone_3d.conv3.register_forward_hook(capture)
        try:
            return super().encode_membrane(lidar, batch_size)
        finally:
            handle.remove()

    def _detect_highres(self, spatial, local):
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))

    def detect_bev_sample(self, spatial, sample_index):
        local = self._local_shallow[sample_index:sample_index+1] if self.use_shallow_skip else None
        return self._detect_highres(spatial, local)

    def detect_bev(self, spatial):
        # The local auxiliary detection is evaluated as a full minibatch.
        return self._detect_highres(spatial, self._local_shallow)

    def forward(self, data):
        try:
            return super().forward(data)
        finally:
            self._local_shallow = None

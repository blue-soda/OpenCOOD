"""E-3DSNN 3D + BEV backbone with an OpenCOOD one-stage detection head.

DAIR adaptation; excludes the paper's VoxelRCNN RoI refinement stage.
"""
import math
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.e3dsnn.backbone_3d import VoxelBackBone8x_3dv_snn
from opencood.models.sub_modules.e3dsnn.backbone_2d import BaseBEVBackbone_spike


class AttributeConfig(dict):
    __getattr__ = dict.__getitem__


class E3dsnnSingle(nn.Module):
    def __init__(self, args):
        super().__init__()
        extent = np.asarray(args['lidar_range'][3:]) - args['lidar_range'][:3]
        grid = np.rint(extent / args['voxel_size']).astype(np.int64)
        # Two BEV scales must align exactly; no implicit crop/interpolation.
        if any(grid[:2] % 16):
            raise ValueError('XY grid must be divisible by 16 for the reference BEV backbone')
        self.backbone_3d = VoxelBackBone8x_3dv_snn({}, 4, grid)
        depth = int(grid[2]) + 1
        for _ in range(3):
            depth = (depth + 1) // 2
        depth = (depth - 3) // 2 + 1
        if depth < 1:
            raise ValueError('Z grid is too shallow for E-3DSNN')
        bev_config = AttributeConfig(LAYER_NUMS=[5, 5], LAYER_STRIDES=[1, 2],
            NUM_FILTERS=[64, 128], UPSAMPLE_STRIDES=[1, 2], NUM_UPSAMPLE_FILTERS=[128, 128])
        self.backbone_2d = BaseBEVBackbone_spike(bev_config, 128 * depth)
        anchors = args['anchor_number']
        self.cls_head = nn.Conv2d(256, anchors, 1)
        self.reg_head = nn.Conv2d(256, anchors * 7, 1)
        self.dir_head = nn.Conv2d(256, anchors * args['dir_args']['num_bins'], 1)
        nn.init.constant_(self.cls_head.bias, -math.log(99.))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)
        nn.init.zeros_(self.reg_head.bias)

    def encode_lidar(self, lidar, batch_size):
        counts = lidar['voxel_num_points'].view(-1, 1).to(lidar['voxel_features'].dtype)
        if not len(counts):
            raise ValueError('Empty single-agent voxel batch')
        mean = lidar['voxel_features'].sum(1) / counts.clamp_min(1)
        batch = self.backbone_3d({'voxel_features': mean,
            'voxel_coords': lidar['voxel_coords'], 'batch_size': int(batch_size)})
        return batch['encoded_spconv_tensor']

    def detect_bev(self, spatial):
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        return {'psm': self.cls_head(bev), 'rm': self.reg_head(bev), 'dm': self.dir_head(bev)}

    def forward(self, data_dict):
        batch_size = data_dict.get('batch_size')
        if batch_size is None:
            batch_size = data_dict['object_bbx_mask'].shape[0]
        encoded = self.encode_lidar(data_dict['processed_lidar'], batch_size)
        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        result = self.detect_bev(dense.view(b, c * d, h, w))
        if data_dict.get('return_spikes', False):
            result['spike_features'] = encoded.features
            result['spike_coords'] = encoded.indices
            result['spike_stride'] = 8
        return result

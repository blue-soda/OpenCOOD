# -*- coding: utf-8 -*-
"""PointPillar single detector with the CoDynTrust ResNet backbone variant."""

import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import (
    ResNetBEVBackbone,
)
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv


class PointPillarCodyntrustSingle(nn.Module):
    """Single-agent PointPillar compatible with CoDynTrust DAIR checkpoints."""

    def __init__(self, args):
        super(PointPillarCodyntrustSingle, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        if args['base_bev_backbone'].get('resnet', False):
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)

        if 'dir_args' in args:
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_number'],
                kernel_size=1)
        else:
            self.use_dir = False

    def forward(self, data_dict):
        processed_lidar = data_dict['processed_lidar']
        batch_dict = {
            'voxel_features': processed_lidar['voxel_features'],
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
        }
        if 'batch_size' in processed_lidar:
            batch_dict['batch_size'] = processed_lidar['batch_size']
        if 'batch_size' in data_dict:
            batch_dict['batch_size'] = data_dict['batch_size']

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        output_dict = {
            'psm': self.cls_head(spatial_features_2d),
            'rm': self.reg_head(spatial_features_2d),
        }

        if self.use_dir:
            output_dict['dm'] = self.dir_head(spatial_features_2d)

        return output_dict

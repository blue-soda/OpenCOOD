"""OpenCOOD adapter for the migrated TraF-Align components.

The original TraF-Align reader/backbone/fusion files are preserved under the
``traf_`` prefix.  This adapter supplies the small amount of runtime metadata
that the original V2X-Seq/V2V4Real data loader provided explicitly, while
accepting OpenCOOD's ``processed_lidar`` dictionary.
"""

import torch
from torch import nn

from opencood.models.traf_pillar_encoder import PillarFeatureNet
from opencood.models.traf_sparseresnet import SparseResNet
from opencood.models.traf_traf_align_fusion import TrafAlign_


class TrafTrafAlign(nn.Module):
    """TraF-Align reader + sparse backbone + alignment fusion adapter."""

    def __init__(self, args):
        super(TrafTrafAlign, self).__init__()
        self.cfg = args.get('traf_cfg', args)
        self.reader = PillarFeatureNet(self.cfg)
        self.backbone = SparseResNet(self.cfg)
        self.fusion_net = TrafAlign_(self.cfg)

        mapping_dim = int(self.cfg['model']['deform']['mapping_dim'])
        self.cls_head = nn.Conv2d(mapping_dim, 2, kernel_size=1)
        self.reg_head = nn.Conv2d(mapping_dim, 14, kernel_size=1)

    def _runtime_data(self, data_dict):
        data = dict(data_dict)
        record_len = data['record_len']
        if not torch.is_tensor(record_len):
            record_len = torch.as_tensor(record_len, dtype=torch.long)
            data['record_len'] = record_len

        batch_size = int(record_len.numel())
        max_cav = int(self.cfg['train_params']['max_cav'])
        if 'time_delays' not in data:
            data['time_delays'] = torch.zeros(
                (batch_size, max_cav), dtype=torch.float32,
                device=record_len.device)
        return data

    def forward(self, data_dict, infer=False):
        data_dict = self._runtime_data(data_dict)
        lidar = data_dict['processed_lidar']
        features = self.reader(lidar)
        features = self.backbone(features, data_dict)
        fused, trajectory, offsets = self.fusion_net(
            features, data_dict, lidar)
        return {
            'psm': self.cls_head(fused),
            'rm': self.reg_head(fused),
            'x_traj': trajectory,
            'x_offset': offsets,
        }

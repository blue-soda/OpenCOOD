"""Local Count4 detail added to an upsampled cooperative ANN BEV feature."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalCountDetail(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260929)
            self.features = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.LeakyReLU(.1),
                nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.LeakyReLU(.1))
            self.output = nn.Conv2d(128, 256, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, local_counts):
        features = self.features(local_counts)
        features = F.layer_norm(features.permute(0, 2, 3, 1), (128,)).permute(0, 3, 1, 2)
        raw = self.output(features)
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite local detail projection')
        return raw.tanh()


def upsample_bev(bev):
    return F.interpolate(bev, scale_factor=2, mode='bilinear', align_corners=False)

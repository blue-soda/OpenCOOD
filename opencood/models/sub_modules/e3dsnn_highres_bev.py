"""Local Count4 detail added to an upsampled cooperative ANN BEV feature."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalCountDetail(nn.Module):
    def __init__(self, hidden_channels=128, second_dilation=1):
        super().__init__()
        if hidden_channels not in (128, 256) or second_dilation not in (1, 2):
            raise ValueError('Registered local detail widths 128/256, dilation 1/2 only')
        self.hidden_channels = hidden_channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260929)
            self.features = nn.Sequential(
                nn.Conv2d(64, hidden_channels, 3, padding=1, bias=False), nn.LeakyReLU(.1),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=second_dilation,
                          dilation=second_dilation, bias=False), nn.LeakyReLU(.1))
            self.output = nn.Conv2d(hidden_channels, 256, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, local_counts):
        features = self.features(local_counts)
        features = F.layer_norm(features.permute(0, 2, 3, 1), (self.hidden_channels,)).permute(0, 3, 1, 2)
        raw = self.output(features)
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite local detail projection')
        return raw.tanh()


def upsample_bev(bev, scale=2):
    return F.interpolate(bev, scale_factor=scale, mode='bilinear', align_corners=False)

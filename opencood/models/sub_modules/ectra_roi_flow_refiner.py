# -*- coding: utf-8 -*-
"""
ROI-level ECTRA refinement for CoBEVFlow-style box flow.

The matcher returns a normalized grid for ``torch.nn.functional.grid_sample``.
This module predicts a small residual grid and a soft trust mask from the
current ego/collaborator BEV features, the coarse ROI mask, and the irregular
time interval.
"""

import torch
import torch.nn as nn


class EctraRoiFlowRefiner(nn.Module):
    def __init__(self, args):
        super(EctraRoiFlowRefiner, self).__init__()
        self.feature_dim = args.get('feature_dim', 64)
        self.hidden_dim = args.get('hidden_dim', 32)
        self.time_scale = args.get('time_scale', 10.0)
        self.residual_range = args.get('residual_range', 2.0)
        self.min_trust = args.get('min_trust', 0.05)

        in_channels = self.feature_dim * 3 + 3
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.flow_head = nn.Conv2d(self.hidden_dim, 2, kernel_size=1)
        self.trust_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    @staticmethod
    def _identity_grid(batch_size, height, width, device, dtype):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1)
        return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward(self, coarse_flow_grid, reserved_mask, features, record_len, time_intervals):
        """
        Parameters
        ----------
        coarse_flow_grid : torch.Tensor
            CoBEVFlow matcher grid, shape (sum_cav, H, W, 2).
        reserved_mask : torch.Tensor
            CoBEVFlow reserved/ROI mask, shape (sum_cav, C, H, W).
        features : torch.Tensor
            Current-frame BEV feature of each CAV, shape (sum_cav, C, H, W).
        record_len : torch.Tensor
            Number of CAVs in each batch item.
        time_intervals : torch.Tensor
            Irregular delay/time interval for each current-frame CAV.
        """
        num_cav, _, height, width = features.shape
        dtype = features.dtype
        device = features.device

        roi_mask = reserved_mask[:, :1].clamp(0, 1).to(dtype=dtype)
        time_intervals = time_intervals.view(-1, 1, 1, 1).to(device=device, dtype=dtype)
        norm_dt = torch.clamp(time_intervals / max(self.time_scale, 1e-6), min=0.0, max=4.0)
        log_dt = torch.log1p(torch.clamp(time_intervals, min=0.0)) / torch.log(
            torch.tensor(self.time_scale + 1.0, device=device, dtype=dtype))

        ego_refs = []
        for batch_feats in self.regroup(features, record_len):
            ego_refs.append(batch_feats[0:1].repeat(batch_feats.shape[0], 1, 1, 1))
        ego_features = torch.cat(ego_refs, dim=0)

        inputs = torch.cat(
            (
                features,
                ego_features,
                torch.abs(features - ego_features),
                roi_mask,
                norm_dt.expand(num_cav, 1, height, width),
                log_dt.expand(num_cav, 1, height, width),
            ),
            dim=1,
        )

        encoded = self.encoder(inputs)
        residual_pixel = torch.tanh(self.flow_head(encoded)) * self.residual_range
        trust = torch.sigmoid(self.trust_head(encoded))
        trust = self.min_trust + (1.0 - self.min_trust) * trust

        scale = torch.tensor(
            [2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)],
            device=device, dtype=dtype).view(1, 1, 1, 2)
        residual_grid = residual_pixel.permute(0, 2, 3, 1).contiguous() * scale
        refined_flow_grid = coarse_flow_grid + residual_grid * roi_mask.permute(0, 2, 3, 1)

        # The ego frame is the target reference. Keep it untouched so the fusion
        # stack never learns to move the anchor feature itself.
        identity = self._identity_grid(num_cav, height, width, device, coarse_flow_grid.dtype)
        ego_mask = torch.zeros(num_cav, 1, 1, 1, device=device, dtype=dtype)
        start = 0
        for cav_num in record_len.tolist():
            ego_mask[start] = 1.0
            start += cav_num
        refined_flow_grid = refined_flow_grid * (1.0 - ego_mask.permute(0, 2, 3, 1)) \
            + identity * ego_mask.permute(0, 2, 3, 1)
        trust = trust * (1.0 - ego_mask) + ego_mask

        soft_mask = reserved_mask * trust.expand_as(reserved_mask)
        aux = {
            'ectra_roi_trust_mean': trust.mean(),
            'ectra_roi_trust_min': trust.min(),
            'ectra_roi_residual_abs_mean': residual_pixel.abs().mean(),
        }
        return refined_flow_grid, soft_mask, aux

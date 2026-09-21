"""Learned local background correspondence and one common left SE(2) correction."""
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import cell_centers, transform_points, se2


def background_points(message, transform, bounds, max_points=768, margin_cells=2):
    geometry = message['geometry']
    foreground = message['foreground'].float()[None, None]
    if margin_cells:
        foreground = F.max_pool2d(foreground, 2*margin_cells+1, stride=1, padding=margin_cells)
    valid = message['observed'] & ~foreground[0, 0].bool()
    indices = torch.where(valid.reshape(-1))[0]
    if len(indices) > max_points:
        indices = indices[torch.linspace(0, len(indices)-1, max_points, device=indices.device).long()]
    xyz = cell_centers(*valid.shape, bounds, geometry).reshape(-1, 3)[indices].clone()
    descriptor = geometry.reshape(-1, geometry.shape[-1])[indices]
    xyz[:, 2] = descriptor[:, 2]
    return dict(xyz=transform_points(xyz, transform.to(xyz)), descriptor=descriptor[:, [1, 3]])


def weighted_fit(source, target, weights):
    """Closed-form planar Procrustes, no SVD gradient at repeated singular values."""
    weights = weights/weights.sum().clamp_min(1e-8)
    a, b = (source*weights[:, None]).sum(0), (target*weights[:, None]).sum(0)
    x, y = source-a, target-b
    cross = (weights*(x[:, 0]*y[:, 1]-x[:, 1]*y[:, 0])).sum()
    dot = (weights*(x*y).sum(-1)).sum()
    angle = torch.atan2(cross, dot+1e-8)
    c, s = angle.cos(), angle.sin()
    rotated = torch.stack((c*a[0]-s*a[1], s*a[0]+c*a[1]))
    return torch.cat((b-rotated, angle[None]))


class BackgroundCalibration(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        h = settings['hidden_dim']
        self.matcher = nn.Sequential(nn.Linear(8, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.matcher[-1].weight)
        nn.init.zeros_(self.matcher[-1].bias)

    def forward(self, source, target):
        xyz, ref = source['xyz'], target['xyz']
        identity = torch.eye(4, device=xyz.device, dtype=xyz.dtype)
        zero = sum(p.sum()*0 for p in self.parameters())
        empty = dict(accepted=False, usable=False, reason='insufficient_background', matches=0,
                     source_points=len(xyz), target_points=len(ref), overlap=0., residual_before=None, residual_after=None)
        if len(xyz) < self.settings['min_matches'] or len(ref) < self.settings['min_matches']:
            return identity, xyz.new_zeros(3)+zero, empty
        diff = ref[None, :, :2]-xyz[:, None, :2]
        distance = diff.square().sum(-1).sqrt()
        height = (ref[None, :, 2]-xyz[:, None, 2]).abs()
        a, b = source['descriptor'], target['descriptor']
        density = (a[:, None, 0]-b[None, :, 0]).abs()/4
        std = (a[:, None, 1]-b[None, :, 1]).abs()/2
        features = torch.stack((distance/self.settings['radius_m'], height/3, density, std,
                                a[:, None, 0].expand_as(distance)/4, b[None, :, 0].expand_as(distance)/4,
                                a[:, None, 1].expand_as(distance)/2, b[None, :, 1].expand_as(distance)/2), -1)
        logits = -(distance/self.settings['temperature_m']).square()-2*height-density-.5*std
        logits = logits+4*self.matcher(features).squeeze(-1).tanh()
        feasible = (distance <= self.settings['radius_m']) & (height <= self.settings['max_height_difference_m'])
        supported = feasible.any(-1)
        probabilities = logits.masked_fill(~feasible, -1e4).softmax(-1)*feasible
        probabilities = probabilities/probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
        correspondences = probabilities@ref[:, :2]
        confidence = probabilities.max(-1).values
        weights = confidence*supported
        count = int(supported.sum())
        overlap = count/max(len(xyz), 1)
        if count < self.settings['min_matches']:
            empty.update(reason='no_overlap', matches=count, overlap=overlap)
            return identity, xyz.new_zeros(3)+zero, empty
        delta = weighted_fit(xyz[:, :2], correspondences, weights)
        raw_transform = se2(delta)
        normalized = weights/weights.sum().clamp_min(1e-8)
        center = (xyz[:, :2]*normalized[:, None]).sum(0)
        centered = xyz[:, :2]-center
        covariance = centered.T@(centered*normalized[:, None])
        spread = float(torch.linalg.eigvalsh(covariance.detach())[0])
        before = ((xyz[:, :2]-correspondences).norm(dim=-1)*normalized).sum()
        after = ((transform_points(xyz, raw_transform)[:, :2]-correspondences).norm(dim=-1)*normalized).sum()
        usable = spread >= self.settings['min_spread_m2'] and overlap >= self.settings['min_overlap']
        finite = bool(torch.isfinite(delta).all())
        bounded = finite and float(delta[:2].norm()) <= self.settings['max_translation_m'] and float(delta[2].abs()) <= self.settings['max_yaw_rad']
        accepted = usable and bounded and float(after) <= self.settings['max_residual_m'] and float(after) <= float(before)+1e-5
        reason = 'accepted' if accepted else ('degenerate' if not usable else ('excessive_correction' if not bounded else 'high_residual'))
        stats = dict(accepted=accepted, usable=usable, reason=reason, matches=count, source_points=len(xyz),
                     target_points=len(ref), overlap=overlap, spread_m2=spread,
                     residual_before=float(before.detach()), residual_after=float(after.detach()),
                     correction_translation_m=float(delta[:2].detach().norm()), correction_yaw_rad=float(delta[2].detach()))
        # The raw estimate is returned for supervised training even if inference falls back.
        return raw_transform if accepted else identity, delta, stats


def pose_loss(delta, correction_gt, settings):
    target_yaw = torch.atan2(correction_gt[1, 0], correction_gt[0, 0])
    translation = F.smooth_l1_loss(delta[:2]/settings['translation_loss_scale_m'],
                                 correction_gt[:2, 3]/settings['translation_loss_scale_m'], reduction='sum')
    angle = (1-torch.cos(delta[2]-target_yaw))/settings['yaw_loss_scale_rad']**2
    return translation+angle

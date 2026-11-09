"""Fine-BEV variants that add capacity through mechanisms, not brute width.

All three keep the sparse encoder and message path untouched, inherit the
R-123/R-117 weights exactly, and only add new parameters downstream of the 2D
BEV feature:

* ``FineBevMultiScale``  fuses the first 2D stage (0.8 m) back into the final
  feature (0.4 m) with a zero-initialised 1x1 projection, so the head sees both
  scales and starts exactly equal to its parent.
* ``FineBevHeadTrunk``   inserts two 3x3 convolutions before the shared heads
  (CenterPoint-style head trunk).
* ``FineBevDecoupled``   gives classification and regression separate 3x3
  adapter features and separate 1x1 heads.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.e3dsnn_snapshot_fine_bev_dilated import E3dsnnSnapshotFineBevDilated
from opencood.models.e3dsnn_snapshot_fine_bev import E3dsnnSnapshotFineBev
from opencood.models.sub_modules.e3dsnn_highres_bev import upsample_bev


class FineBevMultiScale(E3dsnnSnapshotFineBevDilated):
    def __init__(self, args):
        super().__init__(args)
        stage0_ch = self.backbone_2d.blocks[0][2].out_channels
        bev_ch = sum(block[1].out_channels for block in self.backbone_2d.deblocks)
        self.ms_proj = nn.Conv2d(stage0_ch, bev_ch, 1, bias=False)
        nn.init.zeros_(self.ms_proj.weight)
        self._ms_feat = None
        self._ms_handle = self.backbone_2d.blocks[0].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._ms_feat = output

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))


class FineBevHeadTrunk(E3dsnnSnapshotFineBevDilated):
    def __init__(self, args):
        super().__init__(args)
        ch = self.cls_head.in_channels
        self.head_trunk = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

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
        bev = self.head_trunk(bev)
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))


class FineBevDecoupled(E3dsnnSnapshotFineBevDilated):
    def __init__(self, args):
        super().__init__(args)
        ch = self.cls_head.in_channels
        anchors = int(self.cls_head.out_channels)
        bins = int(args['dir_args']['num_bins'])
        self.cls_adapter = nn.Sequential(
            nn.Conv2d(ch, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        self.reg_adapter = nn.Sequential(
            nn.Conv2d(ch, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        self.cls_head = nn.Conv2d(64, anchors, 1)
        self.reg_head = nn.Conv2d(64, anchors * 7, 1)
        self.dir_head = nn.Conv2d(64, anchors * bins, 1)
        nn.init.constant_(self.cls_head.bias, -math.log(99.))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)
        nn.init.zeros_(self.reg_head.bias)

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
        return dict(psm=self.cls_head(self.cls_adapter(bev)),
                    rm=self.reg_head(self.reg_adapter(bev)),
                    dm=self.dir_head(self.reg_adapter(bev)))


class FineBevMultiScaleTrunk(FineBevHeadTrunk):
    """Head trunk plus the zero-initialised multi-scale projection."""

    def __init__(self, args):
        super().__init__(args)
        stage0_ch = self.backbone_2d.blocks[0][2].out_channels
        bev_ch = sum(block[1].out_channels for block in self.backbone_2d.deblocks)
        self.ms_proj = nn.Conv2d(stage0_ch, bev_ch, 1, bias=False)
        nn.init.zeros_(self.ms_proj.weight)
        self._ms_feat = None
        self._ms_handle = self.backbone_2d.blocks[0].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._ms_feat = output

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        bev = self.head_trunk(bev)
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))


class FineBevQuality(FineBevMultiScaleTrunk):
    """Head trunk + a per-anchor IoU quality head folded into the score at eval."""

    def __init__(self, args):
        super().__init__(args)
        ch = self.cls_head.in_channels
        anchors = int(self.cls_head.out_channels)
        self.quality_head = nn.Sequential(
            nn.Conv2d(ch, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, anchors, 1))

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        bev = self.head_trunk(bev)
        psm = self.cls_head(bev)
        qm = self.quality_head(bev)
        if not self.training:
            psm = psm + torch.log(torch.sigmoid(qm) + 1e-6)
        return dict(psm=psm, rm=self.reg_head(bev), dm=self.dir_head(bev), qm=qm)


class FineBevDeepTrunk(FineBevMultiScaleTrunk):
    """Head trunk with a third 3x3 convolution."""

    def __init__(self, args):
        super().__init__(args)
        ch = self.cls_head.in_channels
        self.head_trunk = nn.Sequential(
            *list(self.head_trunk),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )


class FineBevFpn(FineBevMultiScaleTrunk):
    """Adds the first deblock feature as a second multi-scale source."""

    def __init__(self, args):
        super().__init__(args)
        stage0_ch = self.backbone_2d.blocks[0][2].out_channels
        bev_ch = sum(block[1].out_channels for block in self.backbone_2d.deblocks)
        self.fpn_proj = nn.Conv2d(self.backbone_2d.deblocks[0][1].out_channels, bev_ch, 1, bias=False)
        nn.init.zeros_(self.fpn_proj.weight)
        self._fpn_feat = None
        self._fpn_handle = self.backbone_2d.deblocks[0].register_forward_hook(self._fpn_capture)

    def _fpn_capture(self, module, inputs, output):
        self._fpn_feat = output

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        self._fpn_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        fpn = F.interpolate(self._fpn_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms) + self.fpn_proj(fpn)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        bev = self.head_trunk(bev)
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))


class FineBevMembraneQuality(FineBevMultiScaleTrunk):
    """Uses the receiver membrane itself as the detection quality score.

    No learned quality head: the membrane potential (max over channels and z)
    is normalised per frame and mapped through a fixed sigmoid, then folded into
    the classification logit at inference. Zero training; the point is to show
    that the SNN receiver state carries a usable confidence signal.
    """

    membrane_alpha = 2.0

    def __init__(self, args):
        super().__init__(args)
        self._membrane = None

    def encode_membrane(self, lidar, batch_size):
        out = super().encode_membrane(lidar, batch_size)
        if isinstance(out, tuple) and len(out) == 2:
            self._membrane = out[1]
        return out

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        bev = self.head_trunk(bev)
        psm = self.cls_head(bev)
        if not self.training and self._membrane is not None:
            m = self._membrane.float()
            q = m.amax(1).amax(1, keepdim=True)          # (1, 1, y, x)
            q = F.interpolate(q, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            q = q / (q.mean() + 1e-6)
            quality = torch.sigmoid(self.membrane_alpha * (q - 1.0))
            psm = psm + torch.log(quality + 1e-6)
        return dict(psm=psm, rm=self.reg_head(bev), dm=self.dir_head(bev))


class FineBevMembraneQualityLearned(FineBevMultiScaleTrunk):
    """Learned quality readout from the receiver membrane (S1b).

    Same membrane statistic as S1a (max over channels and z), but the mapping to
    a per-anchor quality logit is a learnable 1x1 convolution supervised by the
    positive-anchor IoU. This is the fair test of whether the membrane carries a
    usable confidence signal, as opposed to a fixed mapping.
    """

    def __init__(self, args):
        super().__init__(args)
        anchors = int(self.cls_head.out_channels)
        self.mem_quality = nn.Conv2d(1, anchors, 1)
        self._membrane = None

    def encode_membrane(self, lidar, batch_size):
        out = super().encode_membrane(lidar, batch_size)
        if isinstance(out, tuple) and len(out) == 2:
            self._membrane = out[1]
        return out

    def _detect_highres(self, spatial, local):
        self._ms_feat = None
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        ms = F.interpolate(self._ms_feat, size=bev.shape[-2:], mode='bilinear', align_corners=False)
        bev = bev + self.ms_proj(ms)
        bev = upsample_bev(bev, self.upsample_factor)
        if self.use_shallow_skip:
            if local is None or local.shape[0] != bev.shape[0]:
                raise ValueError('Ego-only shallow count / head-grid mismatch')
            detail = self.local_detail(local)
            if detail.shape[-2:] != bev.shape[-2:]:
                detail = F.interpolate(detail, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            bev = bev + detail
        bev = self.head_trunk(bev)
        psm = self.cls_head(bev)
        if self._membrane is not None:
            m = self._membrane.float()
            q = m.amax(1).amax(1, keepdim=True)
            q = F.interpolate(q, size=bev.shape[-2:], mode='bilinear', align_corners=False)
            q = q / (q.mean() + 1e-6)
            qm = self.mem_quality(q)
        else:
            qm = torch.zeros_like(psm)
        if not self.training:
            psm = psm + torch.log(torch.sigmoid(qm) + 1e-6)
        return dict(psm=psm, rm=self.reg_head(bev), dm=self.dir_head(bev), qm=qm)


class FineBevQualityNorm(FineBevQuality):
    """Quality head whose eval-time folding keeps the mean score scale.

    The plain variant adds ``log sigmoid(q)`` to every logit, which shifts all
    scores down while q is still uncalibrated and therefore interacts with the
    fixed 0.2 score threshold. This variant divides the factor by its per-frame
    mean first, so the ranking changes but the average score scale does not.
    """

    def __init__(self, args):
        super().__init__(args)
        self.fold_mode = str(args.get('quality_fold', 'norm'))
        self.fold_const = float(args.get('quality_fold_const', 0.5))

    def _detect_highres(self, spatial, local):
        out = super()._detect_highres(spatial, local)
        if not self.training:
            qm = out['qm']
            raw_psm = out['psm'] - torch.log(torch.sigmoid(qm) + 1e-6)
            if self.fold_mode == 'off':
                out['psm'] = raw_psm
            elif self.fold_mode == 'const':
                out['psm'] = raw_psm + torch.log(torch.tensor(self.fold_const, device=raw_psm.device))
            else:
                factor = torch.sigmoid(qm)
                factor = factor / (factor.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
                out['psm'] = raw_psm + torch.log(factor + 1e-6)
        return out


class FineBevProbe(FineBevMultiScaleTrunk):
    """Receiver-mode probe on the fine+trunk architecture (AC-3 ablation)."""

    def __init__(self, args):
        super().__init__(args)
        self.receiver_probe = str(args.get('receiver_probe', 'membrane'))
        if self.receiver_probe not in ('membrane', 'prequantized', 'continuous'):
            raise ValueError('Unknown receiver probe')
        self.joint_receiver.snapshot_receive_mode = self.receiver_probe
        if bool(args.get('freeze_encoder', True)):
            prefixes = ('content.expand.', 'joint_receiver.', 'backbone_2d.',
                        'local_detail.', 'cls_head.', 'reg_head.', 'dir_head.')
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith(prefixes))
            self.freeze_encoder = True
            self.train_encoder = False

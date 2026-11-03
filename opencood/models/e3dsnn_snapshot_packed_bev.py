"""Single collaborative path: shallow counts -> packed BEV -> detector.

Space-to-depth preserves all four XY phases before the existing ANN BEV.
This control removes both the sparse deep stages and the detail shortcut;
it retains the conventional internal multiscale BEV decoder.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.e3dsnn_snapshot_branch_probe import E3dsnnSnapshotBranchProbe
from opencood.models.sub_modules.e3dsnn_highres_bev import upsample_bev


class E3dsnnSnapshotPackedBev(E3dsnnSnapshotBranchProbe):
    def __init__(self, args):
        # Construct the common source before deleting modules, so the existing
        # initialization and RNG consumption remain the same as the other arms.
        super().__init__(dict(args, branch_mode='dual'))
        self.branch_mode = 'shallow'  # Reuse the prefix-only encoder/auxiliary path.
        self.backbone_3d.conv4 = None
        self.backbone_3d.conv_out = None
        self.local_detail = None
        self.use_shallow_skip = False
        first = next(m for m in self.backbone_2d.modules() if isinstance(m, nn.Conv2d))
        if first.in_channels != 384:
            raise ValueError('Registered DAIR backend expects 384 input channels')
        with torch.random.fork_rng(devices=[]):
            self.serial_project = nn.Conv2d(256, 384, 1, bias=False)
            # Full-column-rank cyclic copy, preserving nonnegative count scale
            # at initialization. This is not a learned or calibrated threshold.
            with torch.no_grad():
                self.serial_project.weight.zero_()
                for channel in range(384):
                    self.serial_project.weight[channel, channel % 256, 0, 0] = 1.

    def _detect_shallow(self, counts):
        if counts.shape[1] != 64 or any(v % 2 for v in counts.shape[-2:]):
            raise ValueError('Expected even-sized stride-4 Count4 BEV')
        packed = F.pixel_unshuffle(counts, 2)
        spatial = self.serial_project(packed)
        bev = self.backbone_2d({'spatial_features': spatial})['spatial_features_2d']
        bev = upsample_bev(bev)
        if bev.shape[-2:] != counts.shape[-2:]:
            raise ValueError('Packed single-path head grid mismatch')
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))

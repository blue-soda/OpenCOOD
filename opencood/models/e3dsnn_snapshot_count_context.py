"""Matched local/paid-road count context for the preserved shallow baseline.

Only post-readout counts enter the new residual. Road context is the change
in height-max Count4 caused by the already-paid, aligned message. It is not
new road shallow detail, a continuous membrane bypass, or a new wire format.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.models.sub_modules.e3dsnn_highres_bev import upsample_bev


class CountContextResidual(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261101)
            self.mix = nn.Conv2d(192, 32, 3, padding=1, bias=False)
            self.readout = Multispike()
            self.output = nn.Conv2d(32, 256, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, shallow, context):
        # Nearest expansion preserves the integer provenance of coarse counts.
        context = F.interpolate(context, size=shallow.shape[-2:], mode='nearest')
        membrane = self.mix(torch.cat((shallow, context), dim=1) / 4.)
        membrane = F.layer_norm(membrane.permute(0, 2, 3, 1), (32,)).permute(0, 3, 1, 2)
        return self.output(self.readout(membrane)).tanh()


class E3dsnnSnapshotCountContext(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        self.context_mode = args['count_context']['mode']
        if self.context_mode not in ('none', 'local', 'road') or not self.use_shallow_skip:
            raise ValueError('Context study requires the preserved shallow baseline')
        self.count_context = CountContextResidual() if self.context_mode != 'none' else None
        self._context_local = self._context_road = None
        # Diagnosis-only intervention; never changed by ordinary train/eval.
        self.context_intervention = args['count_context'].get('intervention', 'normal')

    def encode_membrane(self, lidar, batch_size):
        sparse, membrane = super().encode_membrane(lidar, batch_size)
        if self.count_context is not None:
            self._context_local = sparse.dense().amax(2)
        return sparse, membrane

    def _observe_counts(self, local_counts, fused_counts):
        self._context_road = fused_counts.amax(2) - local_counts.amax(2)

    def _context_detect(self, spatial, local, context):
        bev = upsample_bev(self.backbone_2d({'spatial_features': spatial})['spatial_features_2d'])
        bev = bev + self.local_detail(local)
        if self.count_context is not None:
            if self.context_intervention == 'zero_context':
                context = torch.zeros_like(context)
            elif self.context_intervention == 'shift_context':
                context = torch.roll(context, shifts=context.shape[-1] // 2, dims=-1)
            elif self.context_intervention not in ('normal', 'zero_branch'):
                raise ValueError('Unknown context diagnostic intervention')
            delta = self.count_context(local, context)
            if self.context_intervention == 'zero_branch':
                delta = torch.zeros_like(delta)
            bev = bev + delta
        return dict(psm=self.cls_head(bev), rm=self.reg_head(bev), dm=self.dir_head(bev))

    def detect_bev_sample(self, spatial, sample_index):
        if self.count_context is None:
            return super().detect_bev_sample(spatial, sample_index)
        local = self._local_shallow[sample_index:sample_index+1]
        context = self._context_local[sample_index:sample_index+1]
        if self.context_mode == 'road':
            context = self._context_road if self._context_road is not None else torch.zeros_like(context)
        try:
            return self._context_detect(spatial, local, context)
        finally:
            self._context_road = None

    def detect_bev(self, spatial):
        if self.count_context is None:
            return super().detect_bev(spatial)
        # Ego auxiliary has no road context, including after a cooperative pass.
        context = self._context_local
        if self.context_mode == 'road':
            context = torch.zeros_like(context)
        return self._context_detect(spatial, self._local_shallow, context)

    def forward(self, data):
        if self.training and self.context_intervention != 'normal':
            raise ValueError('Counterfactual interventions are evaluation-only')
        if self.context_mode == 'road':
            self.joint_receiver.snapshot_readout_observer = self._observe_counts
        try:
            return super().forward(data)
        finally:
            if self.context_mode == 'road':
                del self.joint_receiver.snapshot_readout_observer
            self._context_local = self._context_road = None

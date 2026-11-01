"""Fixed-range count precision probe of the ego-only shallow readout.

No new trainable state or road information. Continuous clipping is an explicitly
non-spiking diagnostic, never a candidate for the SNN architecture claim.
"""
import torch
from torch import nn
from opencood.models.e3dsnn_snapshot_highres import E3dsnnSnapshotHighres


class CountPrecisionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, steps):
        ctx.save_for_backward(x)
        theta = 4. / steps
        return torch.floor(x.clamp(0, 4) / theta + .5).clamp(0, steps) * theta

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * ((x > 0) & (x < 4)).to(grad.dtype), None


class FixedRangeCount(nn.Module):
    def __init__(self, steps):
        super().__init__()
        if steps not in (4, 8, 16):
            raise ValueError('Registered count precision: 4, 8, or 16')
        self.steps = steps

    def forward(self, x):
        return CountPrecisionFunction.apply(x, self.steps)


class ContinuousClipDiagnostic(nn.Module):
    def forward(self, x):
        return x.clamp(0, 4)


class E3dsnnSnapshotPrecision(E3dsnnSnapshotHighres):
    def __init__(self, args):
        super().__init__(args)
        if not self.use_shallow_skip:
            raise ValueError('Precision probe requires the preserved shallow branch')
        self.precision_mode = args['shallow_precision']
        if self.precision_mode == 'continuous':
            self.shallow_readout = ContinuousClipDiagnostic()
        elif self.precision_mode in ('count4', 'count8', 'count16'):
            self.shallow_readout = FixedRangeCount(int(self.precision_mode[5:]))
        else:
            raise ValueError('Unknown shallow readout precision')

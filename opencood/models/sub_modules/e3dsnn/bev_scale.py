"""Bounded per-layer Count4 step size; the underlying count remains 0..4."""
import math
import torch
from torch import nn


class ScaledCount4(nn.Module):
    def __init__(self, neuron, learnable):
        super().__init__()
        self.neuron = neuron
        value = torch.zeros(())
        if learnable:
            self.log_scale = nn.Parameter(value)
        else:
            self.register_buffer('log_scale', value)

    def scale(self):
        return self.log_scale.clamp(-math.log(4.), math.log(4.)).exp()

    def forward(self, x):
        scale = self.scale()
        return scale * self.neuron(x / scale)


def scale_report(module):
    if any(not torch.isfinite(child.log_scale.detach()).all() for child in module.modules()
           if isinstance(child, ScaledCount4)):
        raise ValueError('Nonfinite raw BEV log-scale')
    values = {name: float(child.scale().detach().cpu()) for name, child in module.named_modules()
              if isinstance(child, ScaledCount4)}
    if not values or any(not math.isfinite(v) or v < .25 or v > 4. for v in values.values()):
        raise ValueError('BEV scale audit failed')
    return dict(scales=values, count=len(values), minimum=min(values.values()), maximum=max(values.values()),
                changed_from_one=sum(v != 1. for v in values.values()),
                at_bounds=sum(v <= .25 or v >= 4. for v in values.values()), passed=True)


def add_unit_scales(source, target):
    """Extend a bare source checkpoint only with audited zero log-scales."""
    if not set(source).issubset(target):
        raise ValueError('Source contains unknown state')
    extra = set(target) - set(source)
    if len(extra) != 14 or any(not k.startswith('backbone_2d.') or not k.endswith('.log_scale') for k in extra):
        raise ValueError('Expected exactly fourteen BEV scale entries')
    result = {}
    for key, value in target.items():
        if key in source:
            old = source[key]
            if value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError('Source shape/dtype mismatch: ' + key)
            result[key] = old.clone()
        else:
            if value.ndim != 0 or float(value) != 0.:
                raise ValueError('Scales must start at one')
            result[key] = value.clone()
    return result

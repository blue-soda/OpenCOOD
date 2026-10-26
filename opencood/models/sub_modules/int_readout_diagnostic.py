"""Offline teacher-history counterfactuals; never a deployment cache policy."""
import torch
from opencood.models.sub_modules.int_short_readout import ShortReadoutState
from opencood.models.sub_modules.int_feature_memory import warp_bev


class DiagnosticReadoutState(ShortReadoutState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_targets = True
        self.previous_target = None
        self.observation = None

    @torch.no_grad()
    def step(self, current, state, meta, reset=False):
        value, new, info = super().step(current, state, meta, reset)
        target = info['readout_target']
        self.observation = None
        if not info['readout_refresh']:
            previous = self.previous_target
            if previous is None or previous['stamp'] != state['stamp'] or previous['scene'] != state['scene']:
                raise ValueError('Diagnostic teacher history must follow the same ordered trajectory')
            old, valid = warp_bev(previous['value'], previous['pose'], meta['pose'], self.bounds)
            oracle = self.repair(torch.cat((current, old, valid.to(old.dtype)), 1))
            self.observation = dict(depth=info['readout_depth'], target=target,
                recursive=value, teacher_history=oracle, teacher_warp_only=old)
        # This separate buffer is used only for counterfactual measurement. The
        # returned state remains the real predicted complete readout, unchanged.
        self.previous_target = dict(value=target.detach(), pose=meta['pose'],
            stamp=int(meta['timestamp_us']), scene=meta['segment'])
        return value, new, info


def readout_error(value, target, mask=None):
    if mask is None:
        weight = torch.ones_like(target[:, :1])
    else:
        weight = mask.to(target.dtype)
    count = float(weight.sum()) * target.shape[1]
    return dict(squared_error=float(((value-target).square()*weight).sum()),
        target_energy=float((target.square()*weight).sum()), elements=count)

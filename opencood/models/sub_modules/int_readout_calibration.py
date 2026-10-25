"""Offline scalar calibration; deployed selection remains the ordinary mask."""
import math
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_cache_repair import expand_tiles
from opencood.models.sub_modules.int_budgeted_readout import executed_macs


def threshold_mask(scores,mandatory,threshold):
    changed=expand_tiles(scores>threshold,4,mandatory.shape[-2:])
    return (F.max_pool2d(changed.float(),5,1,2)>0)|mandatory


def fit_threshold(scores,mandatory,w1,w2,ratio=.8,base=.5):
    """Fit aggregate partial-frame MACs on frozen training histories only.

    This is not a per-frame budget guarantee and is not a closed-loop result.
    Batched costs include intermediate halos and protected output cells.
    """
    if (scores.ndim!=4 or scores.shape[1]!=1 or scores.shape[0]==0 or
            mandatory.dtype!=torch.bool or mandatory.shape[:2]!=scores.shape[:2] or
            not torch.isfinite(scores).all() or not 0<ratio<=1 or not math.isfinite(base) or base<0):
        raise ValueError('Finite nonempty training scores and matching mandatory masks required')
    def cost(threshold):
        return sum(executed_macs(threshold_mask(s,m,threshold),w1,w2)
                   for s,m in zip(scores.split(32),mandatory.split(32)))
    baseline=cost(base);target=int(baseline*ratio)
    lo=base;hi=max(base,float(scores.max()))
    floor=cost(hi)
    if floor>target:
        return dict(status='UNREACHABLE',threshold=hi,baseline_macs=baseline,target_macs=target,
                    calibrated_macs=floor,mandatory_macs=floor)
    for _ in range(24):
        mid=(lo+hi)/2
        if cost(mid)<=target:hi=mid
        else:lo=mid
    final=cost(hi)
    assert final<=target
    return dict(status='FITTED',threshold=hi,baseline_macs=baseline,target_macs=target,
                calibrated_macs=final,mandatory_macs=floor,actual_ratio=final/max(baseline,1),
                iterations=24)

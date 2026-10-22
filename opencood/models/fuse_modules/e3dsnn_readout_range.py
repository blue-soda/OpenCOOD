"""Frozen counterfactual: remove only the supported output count ceiling."""
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import count_readout


def paired_count_readouts(local, current, supported):
    if (local.shape != current.shape or supported.shape != (local.shape[0],1)+local.shape[2:]
            or supported.dtype != torch.bool or not torch.isfinite(local).all()
            or not torch.isfinite(current).all()):
        raise ValueError('Invalid shared readout state or spatial mask')
    total=local+current;ego=count_readout(local)
    capped=torch.where(supported,count_readout(total),ego)
    uncapped=torch.where(supported,torch.floor(total.clamp(min=0)+.5),ego)
    return capped,uncapped

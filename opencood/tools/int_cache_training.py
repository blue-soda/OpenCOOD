"""Explicit one-frame cache training, with detached own recurrent values.

The differentiable dense backend writes exactly the same mixed activations as
selected inference. It does not claim sparse training FLOPs or a differentiable
selector. No teacher state is used, and BN remains in evaluation mode.
"""
import torch


def detach_state(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key:detach_state(item) for key,item in value.items()}
    if isinstance(value, list):
        return [detach_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_state(item) for item in value)
    return value


def enable_cache_training(mode):
    stage = mode.spatial.stage
    if not getattr(stage,'compact_projected_state',False) or mode.spatial.execution != 'masked_dense':
        raise ValueError('Only compact masked-dense recurrence supports training')
    if stage.training:
        raise ValueError('BN must remain frozen in eval mode')
    stage.allow_cache_grad = True
    mode.spatial.allow_cache_grad = True
    return mode


def training_step(mode, ego, state, meta):
    prediction, state, info = mode.step(ego,state,meta)
    return prediction, detach_state(state), info

"""Receiver-only readout ablation; sender features and wire format are unchanged."""
import torch


def residual_readout(ego, residual, supported, activation, mode='activated'):
    if mode == 'activated':
        fused = activation(ego + residual)
    elif mode == 'continuous':
        fused = ego + residual
    else:
        raise ValueError('Unknown residual readout: ' + mode)
    return torch.where(supported > 0, fused, ego)

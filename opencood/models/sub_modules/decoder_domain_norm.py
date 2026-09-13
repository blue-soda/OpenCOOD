"""Separate running statistics for shared single-agent/fused decoders."""
import torch
from torch import nn
from torch.nn import functional as F


class DecoderDomainBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, source):
        super().__init__(source.num_features, source.eps, source.momentum,
                         source.affine, source.track_running_stats)
        self.register_buffer('fusion_running_mean', source.running_mean.clone())
        self.register_buffer('fusion_running_var', source.running_var.clone())
        self.register_buffer('fusion_num_batches_tracked', source.num_batches_tracked.clone())
        self.load_state_dict(source.state_dict())
        self.fusion_domain = False
        self.fusion_training = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A single-agent checkpoint has no fused-domain statistics yet.
        for key in ('running_mean', 'running_var', 'num_batches_tracked'):
            if prefix + 'fusion_' + key not in state_dict and prefix + key in state_dict:
                state_dict[prefix + 'fusion_' + key] = state_dict[prefix + key].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, inputs):
        if not self.fusion_domain:
            return super().forward(inputs)
        factor = self.momentum
        if self.fusion_training:
            self.fusion_num_batches_tracked.add_(1)
        if factor is None:
            factor = 1.0 / max(int(self.fusion_num_batches_tracked), 1)
        return F.batch_norm(inputs, self.fusion_running_mean,
                            self.fusion_running_var, self.weight, self.bias,
                            self.fusion_training, factor, self.eps)


def split_decoder_statistics(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DecoderDomainBatchNorm2d(child))
        else:
            split_decoder_statistics(child)


def set_decoder_domain(module, fusion, training):
    for child in module.modules():
        if isinstance(child, DecoderDomainBatchNorm2d):
            child.fusion_domain = fusion
            child.fusion_training = training

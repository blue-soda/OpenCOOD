"""Frozen T8 centered count readout without diagnostic reductions or hooks.

Same dense Conv graph as the validated decoded-count reference. No sparse
execution. Inputs/scales are validated before timing, not via timed reductions.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_if_activation import readout_coded_anchor


class CountOnlyActivation(nn.Module):
    def __init__(self, source):
        super().__init__()
        assert source.mode=='count' and source.steps==8 and source.preserve_tail
        assert source.initial_membrane_units==32768
        self.register_buffer('scales',source.scales.detach().clone())

    def forward(self,x):
        if self.training or torch.is_grad_enabled():
            raise ValueError('Frozen count inference only')
        positive=x.clamp_min(0)
        clipped=torch.minimum(positive,self.scales)
        overflow=positive>self.scales
        drive=torch.floor((clipped/self.scales)*65536).to(torch.int32).clamp_(0,65536)
        drive.masked_fill_(overflow,0)
        counts=torch.floor_divide(drive*8+32768,65536)
        decoded=counts.to(x.dtype)*(self.scales/8)
        return torch.where(overflow,positive,decoded)


class RuntimeCountReadout(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.input_codec=CountOnlyActivation(source.input_codec)
        self.body=copy.deepcopy(source.body)
        self.body.layers[0].double_conv[2]=CountOnlyActivation(source.middle_codec)

    def forward(self,x):
        return self.body(self.input_codec(x))


def runtime_count_anchor(base,scales):
    model=readout_coded_anchor(base,scales,'count',8,True,32768)
    model.core.readout=RuntimeCountReadout(model.core.readout)
    return model.eval()

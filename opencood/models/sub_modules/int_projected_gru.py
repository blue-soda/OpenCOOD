"""Expose cheap low-dimensional gate inputs before local GRU convolutions.

This is a dense architectural baseline, not a sparse execution implementation.
It preserves hidden-state geometry, nonlinearities and spatial dependencies.
"""
import copy
from torch import nn
from opencood.models.sub_modules.int_compact_readout import factorize_input


def projected_gru(source,rank):
    if source.mode!='gru' or rank not in (32,64,128):
        raise ValueError('ANN GRU and rank32/64/128 required')
    result=copy.deepcopy(source).cpu();energy={}
    for name in ('reset_gate_conv','update_gate_conv','candidate_conv'):
        layer=getattr(result.fusion,name)
        if (len(layer)!=4 or not isinstance(layer[0],nn.ZeroPad2d) or
            not isinstance(layer[1],nn.Conv2d) or layer[1].in_channels!=128 or
            layer[1].out_channels!=64 or layer[1].kernel_size!=(3,3) or layer[1].padding!=(0,0)):
            raise ValueError('Unexpected mature GRU layer structure')
        factored,energy[name]=factorize_input(layer[1],rank)
        # Bias-free projection commutes with zero padding. Keep the projection
        # before padding: its map can later be compared in the original lattice.
        layer[0]=factored[0]
        layer[1]=nn.Sequential(nn.ZeroPad2d(1),factored[1])
    return result,energy

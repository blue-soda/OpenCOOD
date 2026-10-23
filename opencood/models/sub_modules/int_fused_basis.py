"""Inference-only fusion of full-rank spatial bases, preserving cache semantics.

The composed kernel replaces, rather than retains, the depthwise+mix weights.
Full refresh uses native convolution; partial frames gather requested 3x3
neighborhoods and evaluate only their output rows. No activation halo is run.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage, SpatialBasisConv
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3_rows, prepared_conv3x3_rows


class FusedBasisConv(nn.Module):
    def __init__(self, source):
        super().__init__()
        if not isinstance(source, SpatialBasisConv):
            raise TypeError('Expected spatial-basis convolution')
        ci, co = source.basis.in_channels, source.mix.out_channels
        rank = source.basis.out_channels // ci
        if rank != 9:
            raise ValueError('Only full-rank basis fusion is enabled')
        self.conv = nn.Conv2d(ci, co, 3, padding=1, bias=source.mix.bias is not None).to(source.mix.weight)
        with torch.no_grad():
            mixing = source.mix.weight.detach().cpu().double().reshape(co, ci, rank)
            basis = source.basis.weight.detach().cpu().double().reshape(ci, rank, 9)
            self.conv.weight.copy_(torch.einsum('oir,irk->oik', mixing, basis).reshape(co, ci, 3, 3))
            if source.mix.bias is not None:
                self.conv.bias.copy_(source.mix.bias)
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return self.conv(x)

    def selected(self, x, mask, plan=None):
        if plan is not None:
            return prepared_conv3x3_rows(self.conv, self.packed_weight, x, plan)
        return selected_conv3x3_rows(self.conv, x, mask)


def fuse_spatial_basis(source, prepared=False):
    if not isinstance(source, SpatialBasisStage) or source.rank != 9 or source.training:
        raise ValueError('Frozen rank9 spatial stage required')
    result = copy.deepcopy(source)
    result.spatial_basis_fused = True
    result.spatial_prepared_indices = prepared
    for block in result.blocks:
        for name in ('conv1', 'conv2'):
            conv = FusedBasisConv(getattr(block, name))
            if prepared:
                # Inference-only derived layout. Explicitly count its storage;
                # no extra learned parameters and never save as trained state.
                conv.register_buffer('packed_weight', conv.conv.weight.detach().permute(0, 2, 3, 1)
                                     .reshape(conv.conv.out_channels, -1).contiguous(), persistent=False)
            setattr(block, name, conv)
    for p in result.parameters(): p.requires_grad_(False)
    return result.eval()

"""Preserve current-E multiscale decoding; factorize only the spatial readout."""
import copy
import torch
from torch import nn


def factorize_input(conv, rank):
    if conv.groups!=1 or conv.stride!=(1,1) or conv.dilation!=(1,1):
        raise ValueError('Only ordinary stride-one readout convolutions supported')
    rank=min(rank,conv.in_channels)
    # W[out,y,x,in] = spatial[out,y,x,rank] @ projection[rank,in].
    w=conv.weight.detach().cpu().double()
    matrix=w.permute(0,2,3,1).reshape(-1,conv.in_channels)
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    projection=nn.Conv2d(conv.in_channels,rank,1,bias=False)
    spatial=nn.Conv2d(rank,conv.out_channels,conv.kernel_size,padding=conv.padding,
                      bias=conv.bias is not None)
    with torch.no_grad():
        projection.weight.copy_(vh[:rank].float().reshape(rank,conv.in_channels,1,1))
        spatial.weight.copy_((u[:,:rank]*s[:rank]).reshape(
            conv.out_channels,*conv.kernel_size,rank).permute(0,3,1,2).float())
        if conv.bias is not None:spatial.bias.copy_(conv.bias.detach().cpu())
    return nn.Sequential(projection,spatial),float(s[:rank].square().sum()/s.square().sum())


def compact_shrink(source,rank):
    result=copy.deepcopy(source).cpu();energy=[]
    for layer in result.layers:
        for index in (0,2):
            layer.double_conv[index],retained=factorize_input(layer.double_conv[index],rank)
            energy.append(retained)
    return result,energy


class CachedFeature(nn.Module):
    """Identity cache read; no zero-weight convolutions in deployed controls."""
    def __init__(self,channels):
        super().__init__();self.channels=channels

    def forward(self,packed):
        return packed[:,128:128+self.channels]


def cached_features():
    return nn.ModuleList([CachedFeature(256),CachedFeature(512)])

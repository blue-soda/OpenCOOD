"""Actual signed, element-sparse 2x2 transposed-convolution execution."""
import torch
from torch import nn


def sparse_deconv_events(conv,delta):
    if torch.is_grad_enabled():raise ValueError('Inference engineering only')
    if not isinstance(conv,nn.ConvTranspose2d) or delta.shape[0]!=1 or conv.kernel_size!=conv.stride or conv.kernel_size!=(2,2) or conv.padding!=(0,0) or conv.output_padding!=(0,0) or conv.dilation!=(1,1) or conv.groups!=1 or conv.bias is not None:
        raise ValueError('Requires bias-free 2x2/stride2/pad0/group1 deconvolution')
    _,cin,h,w=delta.shape;assert cin==conv.in_channels
    coords=torch.nonzero(delta[0]!=0,as_tuple=False);out=delta.new_zeros(conv.out_channels,4*h*w)
    if len(coords):
        channel,y,x=coords.unbind(1);amplitude=delta[0,channel,y,x,None]
        for ky in range(2):
            for kx in range(2):
                # Multiply only weights selected by nonzero input elements.
                contribution=amplitude*conv.weight[channel,:,ky,kx]
                out.index_add_(1,(2*y+ky)*(2*w)+2*x+kx,contribution.t())
    return out.view(1,conv.out_channels,2*h,2*w),dict(active_elements=len(coords),elements=delta.numel(),
        weighted_event_macs=len(coords)*conv.out_channels*4,dense_macs=delta.numel()*conv.out_channels*4,
        scope='Actual element-weight products and scatter-add; dense temporary output and indexing/weight gathers remain overhead.')

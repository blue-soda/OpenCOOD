"""Exact channel-event Conv delta backend; inference-only PyTorch prototype."""
import torch
from torch import nn


def channel_event_delta(conv,delta,chunk=32768):
    if torch.is_grad_enabled():raise ValueError('Inference only')
    if not isinstance(conv,nn.Conv2d) or conv.groups!=1 or conv.stride!=(1,1) or conv.dilation!=(1,1) or conv.kernel_size not in ((1,1),(3,3)) or conv.padding!=(conv.kernel_size[0]//2,)*2 or conv.padding_mode!='zeros':raise ValueError('Same-grid ungrouped Conv required')
    if delta.shape[0]!=1 or delta.shape[1]!=conv.in_channels or type(chunk) is not int or chunk<=0:raise ValueError('Invalid input or chunk size')
    _,_,height,width=delta.shape;coords=torch.nonzero(delta[0]!=0,as_tuple=False);out=delta.new_zeros(conv.out_channels,height*width);deliveries=0
    kernel=conv.kernel_size[0];pad=kernel//2
    for ky in range(kernel):
        for kx in range(kernel):
            y=coords[:,1]+pad-ky;x=coords[:,2]+pad-kx;keep=(y>=0)&(y<height)&(x>=0)&(x<width);selected=coords[keep];target=y[keep]*width+x[keep];deliveries+=len(selected)
            for start in range(0,len(selected),chunk):
                block=selected[start:start+chunk];values=delta[0,block[:,0],block[:,1],block[:,2]]
                weights=conv.weight[:,block[:,0],ky,kx];contribution=weights*values.unsqueeze(0)
                out.index_add_(1,target[start:start+chunk],contribution)
    return out.reshape(1,conv.out_channels,height,width),dict(active_channel_elements=len(coords),scalar_deliveries=deliveries,executed_macs=deliveries*conv.out_channels,dense_macs=height*width*conv.in_channels*conv.out_channels*kernel*kernel,chunk_events=chunk)

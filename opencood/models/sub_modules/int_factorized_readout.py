"""Parent-initialized compact readout, recomputed from current features each frame."""
import copy
import torch
from torch import nn


def factorize_conv(conv,rank):
    if not isinstance(conv,nn.Conv2d) or conv.groups!=1 or conv.padding_mode!='zeros':raise ValueError('Ungrouped zero-padded Conv2d required')
    maximum=min(conv.out_channels,conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1])
    if type(rank) is not int or not 1<=rank<=maximum:raise ValueError('Invalid rank')
    weight=conv.weight.detach().flatten(1);u,s,vh=torch.linalg.svd(weight,full_matrices=False)
    reduce=nn.Conv2d(conv.in_channels,rank,conv.kernel_size,conv.stride,conv.padding,conv.dilation,bias=False).to(weight.device,weight.dtype)
    expand=nn.Conv2d(rank,conv.out_channels,1,bias=conv.bias is not None).to(weight.device,weight.dtype)
    with torch.no_grad():
        reduce.weight.copy_(vh[:rank].reshape_as(reduce.weight));expand.weight.copy_((u[:,:rank]*s[:rank]).reshape_as(expand.weight))
        if conv.bias is not None:expand.bias.copy_(conv.bias)
    return nn.Sequential(reduce,expand)


def compact_readout(source,rank=None):
    result=copy.deepcopy(source);changed=[]
    def visit(module,prefix=''):
        for name,child in list(module.named_children()):
            path=prefix+name
            if isinstance(child,nn.Conv2d) and child.kernel_size==(3,3):
                r=child.out_channels if rank is None else rank
                setattr(module,name,factorize_conv(child,r));changed.append(dict(path=path,input=child.in_channels,output=child.out_channels,rank=r))
            else:visit(child,path+'.')
    visit(result)
    if not changed:raise ValueError('No spatial readout convolutions found')
    return result,changed

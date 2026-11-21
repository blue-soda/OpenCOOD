"""Exact fixed-grid reuse of the deep input contribution before fusion nonlinearity."""
import torch
from torch import nn
from torch.nn import functional as F


class ExactFusionCache(nn.Module):
    def __init__(self,conv,block=256):
        super().__init__()
        if (conv.in_channels!=384 or conv.out_channels!=128 or conv.kernel_size!=(3,3)
                or conv.stride!=(1,1) or conv.padding!=(1,1) or conv.groups!=1 or conv.dilation!=(1,1)):
            raise ValueError('Expected the first rank-128 fusion convolution')
        self.weight=conv.weight
        self.bias=conv.bias
        self.block=block
        self.previous=None

    def begin(self,previous,refresh,local):
        self.previous=None if refresh is not None or previous is None else previous.get('fusion_cache')
        self.local=local
        self.new=None

    def forward(self,x):
        c=128
        h,w=x.shape[-2:]
        deep=x[:,256:]
        first=F.conv2d(x[:,:256],self.weight[:,:256],self.bias,padding=1)
        if self.local and self.previous is not None:
            prev_input,old=self.previous
            changed=(deep!=prev_input).any(dim=1,keepdim=True)
            dirty=F.max_pool2d(changed.float(),3,1,1).flatten()!=0
            ids=dirty.nonzero().flatten()
            output=old.clone()
        else:
            ids=torch.arange(h*w,device=x.device)
            output=x.new_empty(h*w,c)
        padding,mac,extra=self.compute_deep(deep,ids,output)
        self.new=(deep.detach().clone(),output)
        self.previous=None
        self.ledger=dict(sites=h*w,dirty=len(ids),executed=len(ids)+padding,padding=padding,
                         float_conv=2*(h*w*256*9*c+mac),
                         merge_add=h*w*c,cache_bytes=(deep.numel()+output.numel())*deep.element_size(),**extra)
        return first+output.T.reshape(1,c,h,w)

    def compute_deep(self,deep,ids,output):
        _,c,h,w=deep.shape
        size=len(ids)
        padding=(-size)%self.block
        if size:
            # A fixed GEMM shape gives the dense reference and selected sites
            # the same floating accumulation order. Padding jobs are charged.
            source=F.pad(deep[0],(1,1,1,1)).permute(1,2,0).contiguous().view(-1,c)
            weights=self.weight[:,256:].contiguous().view(c,-1)
            axis=torch.arange(3,device=deep.device)
            offsets=(axis[:,None]*(w+2)+axis).flatten()
            padded=F.pad(ids,(0,padding),value=int(ids[-1]))
            for start in range(0,len(padded),self.block):
                selected=padded[start:start+self.block]
                origin=(selected//w)*(w+2)+selected%w
                patch=source[origin[:,None]+offsets].transpose(1,2).reshape(self.block,-1)
                result=F.linear(patch,weights)
                used=min(self.block,size-start)
                output[selected[:used]]=result[:used]
        return padding,(size+padding)*9*c*c,{}


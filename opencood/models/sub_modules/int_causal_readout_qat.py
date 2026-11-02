"""Fixed-grid readout QAT whose forward matches bounded integer reference.

The straight-through derivatives are surrogate gradients. BN statistics remain
frozen. FP64 convolution is a reference/training path, not a speed claim.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_readout_events import ChangeEncodedReadout


class CausalReadoutQAT(ChangeEncodedReadout):
    def __init__(self,source,quanta,enabled):
        super().__init__(source,quanta);self.enabled=bool(enabled);self.last_reference_flops=0;self.last_bounds=[]

    def forward(self,x):
        if self.training:raise ValueError('BN statistics must remain frozen')
        self.last_reference_flops=0;self.last_bounds=[];i=0
        for layer in self.layers:
            if not self.enabled or not isinstance(layer,nn.Conv2d):x=layer(x);continue
            quantum=getattr(self,'quantum_%d'%i).double();normalized=x.double()/quantum;rounded=torch.round(normalized).detach()
            if not bool(torch.isfinite(rounded).all()) or float(rounded.abs().max())>=2**31:raise ValueError('Input integer range exceeded')
            inputs=rounded+(normalized-normalized.detach())
            folded=layer.weight.double()*quantum.reshape(1,-1,1,1);scale=folded.detach().abs().flatten(1).amax(1).clamp_min(1e-12)/32767
            weight_normalized=folded/scale.reshape(-1,1,1,1);weight_rounded=torch.round(weight_normalized).detach();weight=weight_rounded+(weight_normalized-weight_normalized.detach())
            bound=int(weight_rounded.abs().flatten(1).sum(1).max())*int(rounded.abs().max())
            if bound>=2**53:raise ValueError('FP64 exact accumulation bound exceeded')
            acc=F.conv2d(inputs,weight,None,layer.stride,layer.padding)
            if not torch.equal(acc.detach(),acc.detach().round()):raise RuntimeError('QAT forward lost integer exactness')
            self.last_bounds.append(bound);self.last_reference_flops+=2*x.shape[-2]*x.shape[-1]*layer.in_channels*layer.out_channels*layer.kernel_size[0]**2
            x=(acc*scale.reshape(1,-1,1,1)).to(x.dtype)
            if layer.bias is not None:x=x+layer.bias.reshape(1,-1,1,1)
            i+=1
        return x

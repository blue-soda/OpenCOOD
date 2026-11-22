"""FP32 128-channel observation convolution with optional exact fixed-grid reuse."""
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionKernel


class NonzeroObservationConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        if (original.in_channels,original.out_channels,original.kernel_size,original.stride,
                original.padding,original.dilation,original.groups,original.padding_mode) != (
                128,128,(3,3),(1,1),(1,1),(1,1),1,'zeros'):
            raise ValueError('Expected a 128-channel stride-one 3x3 convolution')
        self.register_buffer('packed_weights', original.weight.detach().permute(2,3,1,0).contiguous())
        self.bias = original.bias
        self.kernel = NonzeroFusionKernel()
        self.previous = None

    def begin(self, reuse, refresh):
        self.reuse = reuse
        if refresh:
            self.previous = None

    def forward(self, x):
        if torch.is_grad_enabled() or x.dtype != torch.float32 or x.shape[:2] != (1,128):
            raise ValueError('Frozen single FP32 stream required')
        h,w = x.shape[2:]
        if self.reuse and self.previous is not None:
            old, prior = self.previous
            dirty = F.max_pool2d((x != old).any(1,keepdim=True).float(),3,1,1).bool()
            sites = dirty.flatten().nonzero().flatten()
            buffer = prior.clone()
        else:
            sites = torch.arange(h*w,device=x.device)
            buffer = x.new_empty((h*w,128))
        taps = self.kernel.run(x,self.packed_weights,sites,buffer)
        self.visits = dict(mac=taps*128,sites=len(sites),full_sites=h*w)
        if self.reuse:
            self.previous = (x.clone(),buffer)
        out = buffer.T.reshape(1,128,h,w).contiguous()
        if self.bias is not None:
            out = out+self.bias.view(1,-1,1,1)
        return out

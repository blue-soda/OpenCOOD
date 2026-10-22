"""Dense cheap spatial context, selected expensive per-cell state updates.

The retained mature stem and decoder are a precision prior, not a guarantee:
this new update has to be distilled and validated as a dense baseline first.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class CellNorm(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.LayerNorm(channels)

    def forward(self,x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()


class ContextualPointwiseStage(nn.Module):
    """128-channel half-resolution H; pointwise update has radius zero.

    S is the current mature first residual block. Two cheap depthwise filters
    carry local and wider context into the update, and always execute densely.
    The expensive MLP consumes [S, localS, wideS, alignedH, valid]. The retained
    cache is the complete output H, not the MLP residual or its gate logits.
    """
    def __init__(self,source_stage):
        super().__init__()
        if len(source_stage)!=3 or source_stage[0].conv2.out_channels!=128:
            raise ValueError('Expected the mature128-channel first stage')
        self.stem=copy.deepcopy(source_stage[0])
        self.local=nn.Conv2d(128,128,3,padding=1,groups=128,bias=False)
        self.wide=nn.Conv2d(128,128,9,padding=4,groups=128,bias=False)
        self.update=nn.Sequential(nn.Conv2d(513,512,1,bias=False),CellNorm(512),nn.SiLU(),
            nn.Conv2d(512,512,1,bias=False),CellNorm(512),nn.SiLU(),nn.Conv2d(512,128,1))
        with torch.no_grad():
            self.local.weight.fill_(1/9.);self.wide.weight.fill_(1/81.)
            self.update[0].weight[:,384:].zero_()
            self.update[-1].weight.zero_();self.update[-1].bias.zero_()
        for p in self.stem.parameters():p.requires_grad_(False)
        self.train(False)

    def train(self,mode=True):
        super().train(mode);self.stem.eval();return self

    def observe(self,temporal):
        current=self.stem(temporal)
        return current,torch.cat((current,self.local(current),self.wide(current)),1)

    def pack(self,evidence,old,valid):
        return torch.cat((evidence,old,valid.to(evidence.dtype)),1)

    def forward(self,temporal):
        # Stage A: dense current-only baseline, no synthetic teacher history.
        current,evidence=self.observe(temporal)
        packed=self.pack(evidence,torch.zeros_like(current),torch.zeros_like(current[:,:1]))
        return F.relu(current+self.update(packed))

    def selected_update(self,current,evidence,aligned,valid,mask):
        if self.training or torch.is_grad_enabled():raise ValueError('Frozen selected inference only')
        h,w=current.shape[-2:]
        if current.shape[0]!=1 or mask.shape!=(1,1,h,w) or mask.dtype!=torch.bool:
            raise ValueError('Expected a matching single-frame boolean mask')
        selected=mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if not selected.numel():return aligned.clone()
        packed=self.pack(evidence,aligned,valid)
        # Each spatial column is independent: no neighboring outputs or halo
        # activations are calculated by the expensive pointwise network.
        columns=packed.reshape(1,513,-1)[:,:,selected].unsqueeze(2)
        residual=self.update(columns).squeeze(2)
        value=F.relu(current.reshape(1,128,-1)[:,:,selected]+residual)
        output=aligned.reshape(1,128,-1).clone();output[:,:,selected]=value
        return output.reshape(1,128,h,w)

    def linear_selected_update(self,current,evidence,aligned,valid,mask):
        """Same frozen update in row-major layout, without per-layer transposes.

        Keep original parameters/state_dict and the training forward unchanged.
        Explicit MACs count the executed functional linear layers, which Conv2d
        hooks cannot see. Current observation and cache management are separate.
        """
        if self.training or torch.is_grad_enabled():raise ValueError('Frozen selected inference only')
        h,w=current.shape[-2:]
        if current.shape[0]!=1 or mask.shape!=(1,1,h,w) or mask.dtype!=torch.bool:
            raise ValueError('Expected a matching single-frame boolean mask')
        self.last_selected_linear_macs=0
        selected=mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if not selected.numel():return aligned.clone()
        packed=self.pack(evidence,aligned,valid)
        values=packed.reshape(513,-1)[:,selected].transpose(0,1).contiguous()
        for layer in self.update:
            if isinstance(layer,nn.Conv2d):
                assert layer.kernel_size==(1,1) and layer.groups==1
                values=F.linear(values,layer.weight[:,:,0,0],layer.bias)
                self.last_selected_linear_macs+=selected.numel()*layer.weight.numel()
            elif isinstance(layer,CellNorm):values=layer.norm(values)
            else:values=layer(values)
        value=F.relu(current.reshape(128,-1)[:,selected]+values.transpose(0,1))
        output=aligned.reshape(128,-1).clone();output[:,selected]=value
        return output.reshape(1,128,h,w)

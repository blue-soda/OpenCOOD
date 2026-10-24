"""Dense precision gate for a shorter, locally recomputable spatial update.

Retain mature stem and first residual block; distill the original two-block H
into one block. Optional cheap current context restores its nominal receptive
field before the two expensive layers. Spatial reuse is deliberately disabled
until dense accuracy is assessed; FM/deep caches keep their established rules.
"""
import copy
import torch
from torch import nn


class ShortSpatialStage(nn.Module):
    def __init__(self,source,context=False):
        super().__init__()
        if len(source)!=3 or source[1].downsample is not None:
            raise ValueError('Expected mature stem and two identity residual blocks')
        self.stem=copy.deepcopy(source[0])
        self.block=copy.deepcopy(source[1])
        self.use_context=context
        channels=self.block.conv1.in_channels
        self.context=None
        if context:
            self.context=nn.Sequential(nn.Conv2d(channels,channels,5,padding=2,groups=channels,bias=False),
                nn.Conv2d(channels,channels,1,bias=False)).to(self.block.conv1.weight)
            # Same initial function in both arms; depthwise starts as identity.
            with torch.no_grad():
                self.context[0].weight.zero_();self.context[0].weight[:,:,2,2]=1.
                self.context[1].weight.zero_()
        for p in self.parameters():p.requires_grad_(False)
        for conv in (self.block.conv1,self.block.conv2):
            for p in conv.parameters():p.requires_grad_(True)
        if self.context is not None:
            for p in self.context.parameters():p.requires_grad_(True)
        self.eval()

    def observe(self,x):
        current=self.stem(x)
        return current if self.context is None else current+self.context(current)

    def forward(self,x):
        return self.block(self.observe(x))

    def enable_update_bn_affine(self):
        """Adapt only the retained update block's affine BN; keep stats fixed.

        This preserves the initial function and inference operators. Frozen
        running statistics also preserve per-location evaluation for future
        selected-region execution, unlike training with batch statistics.
        """
        for module in self.block.modules():
            if isinstance(module,nn.BatchNorm2d):
                for p in module.parameters():p.requires_grad_(True)
                module.eval()

    def train(self,mode=True):
        super().train(mode)
        self.stem.eval()
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self

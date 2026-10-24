"""One persistent spatial H; recompute intermediate supports from current E.

Dependency-complete local updates do not consume old intermediate values at
their final core. Zero-filled temporary canvases replace those three caches.
Their allocation/compute is real and must remain included in measured latency.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_dependency_spatial_state import DependencyProjectedSpatialStage


class SingleSpatialStateStage(DependencyProjectedSpatialStage):
    def __init__(self,source):
        super().__init__(source)
        self.single_spatial_state=True
        self.cache_channels=[self.dependency_cache_channels[-1]]

    def dense_from_stem(self,current):
        for block in self.blocks:
            first=F.relu(block.bn1(block.conv1(current)))
            projected=block.conv2.project(first)
            current=F.relu(block.bn2(block.conv2.mix(projected))+current)
        return current,[current]

    def selected_from_stem(self,current,aligned_maps,mask,execution='selected'):
        if self.training or torch.is_grad_enabled():raise ValueError('Frozen single-state inference required')
        if len(aligned_maps)!=1:raise ValueError('Exactly one complete historical H required')
        old=aligned_maps[0]
        if old.shape!=(1,self.cache_channels[0],*current.shape[-2:]) or old.dtype!=current.dtype or old.device!=current.device:
            raise ValueError('Incompatible final spatial state')
        temporary=[current.new_zeros(1,c,*current.shape[-2:]) for c in self.dependency_cache_channels[:-1]]
        output,_,macs=super().selected_from_stem(current,temporary+[old],mask,execution)
        return output,[output],macs

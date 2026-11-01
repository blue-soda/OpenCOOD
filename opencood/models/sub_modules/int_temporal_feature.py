"""Frozen pyramid integration of real-frame feature memory; identity at init."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_spike_memory import INTSpikeMemory
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_block_cache import block_lidar_range


def residual_feature_memory(mode,channels,bounds,memory_channels=32):
    memory=INTSpikeMemory(mode,channels,bounds,memory_channels=memory_channels,
        tau_u=.2,tau_r=.1,threshold=1.,input_gain=.25,max_gap_s=.25)
    nn.init.zeros_(memory.readout.weight)
    return memory


class TemporalFeatureAnchor(AnchoredLatePyramidDetector):
    """Inference engineering only; parent deliberately forbids training/grad."""
    def __init__(self,base,mode,reset_memory=False):
        super().__init__(base)
        self.core=copy.deepcopy(base.core)
        self.core.temporal_feature=residual_feature_memory(mode,256,self.bounds)
        self.reset_feature_memory=reset_memory
        self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        previous=None if state is None else state.get('feature_memory')
        memory=self.core.temporal_feature;results=[]
        def inject(module,inputs,coarse):
            memory.lidar_range=block_lidar_range(self.bounds,spatial.shape[-2:],coarse.shape[-2:],4)
            delta,new,info=memory.step(coarse,previous,meta,
                reset=self.reset_feature_memory or bool(meta.get('reset_before',False)),
                detach_state=detach_state)
            results.append((new,info))
            return coarse+delta
        hook=self.core.coarse_update.register_forward_hook(inject)
        try:prediction,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        finally:hook.remove()
        assert len(results)==1
        new['feature_memory'],info['feature_memory']=results[0]
        extra=info['feature_memory']['state_bytes']
        info['state_bytes']+=extra;info['cache_bytes']+=extra
        return prediction,new,info


def detach_temporal_state(value):
    """Detach all nested graph tensors, preserving cached values and metadata."""
    if torch.is_tensor(value):return value.detach()
    if isinstance(value,dict):return {k:detach_temporal_state(v) for k,v in value.items()}
    if isinstance(value,list):return [detach_temporal_state(v) for v in value]
    if isinstance(value,tuple):return tuple(detach_temporal_state(v) for v in value)
    return value


class DifferentiableTemporalFeatureAnchor(TemporalFeatureAnchor):
    """Adapter gradients in eval mode; frozen spatial parameters and BN stats."""
    differentiable_state=True

    def __init__(self,base,mode,reset_memory=False):
        super().__init__(base,mode,reset_memory)
        for p in self.parameters():p.requires_grad_(False)
        for p in self.core.temporal_feature.parameters():p.requires_grad_(True)

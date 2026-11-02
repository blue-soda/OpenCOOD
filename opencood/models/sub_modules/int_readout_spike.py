"""Real-frame readout membrane with FP32 dynamics and optional FP16 storage."""
import copy
import torch
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_spatial_spike import SpatialActivationMemory
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class ReadoutSpikeAnchor(AnchoredLatePyramidDetector):
    differentiable_state=True
    trainable_names=tuple('core.readout.layers.0.double_conv.%d.%s'%(i,n)
                         for i in (0,3) for n in ('weight','bias'))

    def __init__(self,base,mode,scale,storage_dtype=torch.float32,reset_memory=False):
        super().__init__(base);self.core=copy.deepcopy(base.core)
        self.activation_memory=SpatialActivationMemory(mode,scale,self.bounds,storage_dtype=storage_dtype)
        self.reset_activation_memory=reset_memory
        layers=self.core.readout.layers[0].double_conv
        for i,c in ((0,384),(3,256)):
            m=layers[i]
            if (m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups)!=(c,256,(3,3),(1,1),(1,1),(1,1),1):
                raise ValueError('Unexpected readout convolution')
            if m.bias is None:raise ValueError('Expected original readout bias')
        for n,p in self.named_parameters():p.requires_grad_(n in self.trainable_names)
        self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        previous=None if state is None else state.get('readout_membrane');result=[]
        def replace(module,inputs):
            a=inputs[0];self.last_input=a
            self.activation_memory.bounds=block_lidar_range(self.bounds,spatial.shape[-2:],a.shape[-2:],2)
            output,new,info=self.activation_memory.step(a,previous,meta,
                reset=self.reset_activation_memory or bool(meta.get('reset_before',False)),detach_state=detach_state)
            self.last_output=output;result.append((new,info));return (output,)
        def observe(module,inputs):self.last_common_input=inputs[0]
        layers=self.core.readout.layers[0].double_conv
        hooks=[layers[0].register_forward_pre_hook(observe),layers[3].register_forward_pre_hook(replace)]
        try:pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        finally:
            for hook in hooks:hook.remove()
        assert len(result)==1
        new['readout_membrane'],info['readout_activation']=result[0]
        extra=info['readout_activation']['state_bytes'];info['state_bytes']+=extra;info['cache_bytes']+=extra
        return pred,new,info

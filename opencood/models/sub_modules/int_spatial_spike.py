"""Replace one existing spatial-convolution input with aligned real-frame dynamics."""
import copy
import math
import torch
from torch import nn
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_spike_memory import SurrogateSpike
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_block_cache import block_lidar_range


def deliveries3x3(active,out_channels):
    """Exact valid stride1/pad1 3x3 fanout; omit multiplications into padding."""
    h,w=active.shape[-2:]
    y=torch.full((h,),3,device=active.device,dtype=torch.int64);y[0]-=1;y[-1]-=1
    x=torch.full((w,),3,device=active.device,dtype=torch.int64);x[0]-=1;x[-1]-=1
    return int((active.to(torch.int64)*y[:,None]*x[None,:]).sum())*out_channels


class SpatialActivationMemory(nn.Module):
    def __init__(self,mode,scale,bounds,tau=.2):
        super().__init__()
        if mode not in ('ann','lif','leaky') or not math.isfinite(tau) or tau<=0:raise ValueError('Invalid dynamics')
        scale=torch.as_tensor(scale,dtype=torch.float32).reshape(1,-1,1,1)
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):raise ValueError('Invalid scale')
        self.register_buffer('scale',scale.clone());self.mode=mode;self.bounds=bounds;self.tau=tau

    def step(self,a,state,meta,reset=False,detach_state=True):
        if a.shape[0]!=1 or a.shape[1]!=self.scale.shape[1]:raise ValueError('One matching stream required')
        if self.mode=='ann':return a,None,dict(state_bytes=0,reset=None,warp=False)
        reason='initial' if state is None else None;dt=0.
        if state is not None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                dt=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
                if dt<=0:raise ValueError('Noncausal activation state')
                if dt>.25:reason='gap'
        if reset:reason='explicit'
        if reason is None:past,_=warp_bev(state['value'],state['pose'],meta['pose'],self.bounds)
        else:past=torch.zeros_like(a)
        v=math.exp(-dt/self.tau)*past+a/self.scale
        if self.mode=='lif':emission=SurrogateSpike.apply(v-1.);u=v-emission
        else:emission=torch.relu(v);u=v
        output=self.scale*emission
        with torch.no_grad():
            if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(output).all()):raise RuntimeError('Nonfinite activation memory')
            info=dict(state_bytes=u.numel()*u.element_size(),reset=reason,warp=reason is None,delta_t=dt,
                emission_nonzero_fraction=float((emission!=0).float().mean()),emission_mean=float(emission.mean()),
                emission_nonzero_per_channel=(emission!=0).float().mean((0,2,3)).cpu().tolist(),
                membrane_abs_max=float(u.abs().max()),membrane_ge_threshold_fraction=float((u>=1).float().mean()),
                membrane_abs_max_per_channel=u.abs().amax((0,2,3)).cpu().tolist())
        new=dict(value=u.detach() if detach_state else u,scene=meta['segment'],pose=copy.deepcopy(meta['pose']),timestamp_us=int(meta['timestamp_us']))
        return output,new,info


class SpatialSpikeAnchor(AnchoredLatePyramidDetector):
    differentiable_state=True

    def __init__(self,base,mode,scale,reset_memory=False):
        super().__init__(base);self.core=copy.deepcopy(base.core)
        self.activation_memory=SpatialActivationMemory(mode,scale,self.bounds)
        self.reset_activation_memory=reset_memory
        target=self.core.coarse_encoder[3].conv2
        if target.in_channels!=256 or target.out_channels!=256 or target.kernel_size!=(3,3) or target.stride!=(1,1):raise ValueError('Unexpected target convolution')
        for p in self.parameters():p.requires_grad_(False)
        target.weight.requires_grad_(True);self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        previous=None if state is None else state.get('spatial_membrane');result=[]
        def replace(module,inputs):
            a=inputs[0];self.last_input=a
            self.activation_memory.bounds=block_lidar_range(self.bounds,spatial.shape[-2:],a.shape[-2:],4)
            output,new,info=self.activation_memory.step(a,previous,meta,
                reset=self.reset_activation_memory or bool(meta.get('reset_before',False)),detach_state=detach_state)
            self.last_output=output;result.append((new,info))
            return (output,)
        hook=self.core.coarse_encoder[3].conv2.register_forward_pre_hook(replace)
        try:pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        finally:hook.remove()
        assert len(result)==1
        new['spatial_membrane'],info['spatial_activation']=result[0]
        extra=info['spatial_activation']['state_bytes'];info['state_bytes']+=extra;info['cache_bytes']+=extra
        return pred,new,info

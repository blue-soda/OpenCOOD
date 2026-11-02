"""Frozen amplitude-preserving counts; no surrogate training is implied."""
import copy
import math
import torch
from torch import nn
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.models.sub_modules.int_feature_memory import warp_bev


def integer_delivery_statistics(counts,out_channels=256):
    """Exact event multiplicity and bitplane deliveries, including padded edges."""
    if counts.dtype!=torch.int64 or bool((counts<0).any()):raise ValueError('Nonnegative integer counts required')
    maximum=int(counts.max());h,w=counts.shape[-2:]
    if maximum>=2**24:raise ValueError('Counts exceed exact FP32 integer range')
    y=torch.full((h,),3,device=counts.device,dtype=torch.int64);y[0]-=1;y[-1]-=1
    x=torch.full((w,),3,device=counts.device,dtype=torch.int64);x[0]-=1;x[-1]-=1
    fanout=y[:,None]*x[None,:];hist={}
    for f in sorted(int(v) for v in torch.unique(fanout)):
        hist[str(f)]=torch.bincount(counts[...,fanout==f].reshape(-1),minlength=maximum+1).cpu().tolist()
    unary=int((counts*fanout).sum())*out_channels
    sparse=int(((counts!=0)*fanout).sum())*out_channels
    planes=[int((((counts>>bit)&1)*fanout).sum())*out_channels for bit in range(maximum.bit_length())]
    assert unary==sum(value*(1<<bit) for bit,value in enumerate(planes))
    return dict(count_max=maximum,count_sum=int(counts.sum()),elements=counts.numel(),
        nonzero=int(torch.count_nonzero(counts)),count_histogram_by_fanout=hist,
        unary_event_acs=unary,nonzero_sparse_macs=sparse,bitplane_acs=planes,bitplane_additions=sum(planes))


class AmplitudeMemory(nn.Module):
    def __init__(self,mode,scale,bounds,storage_dtype=torch.float32,tau=.2):
        super().__init__()
        if mode not in ('ann','count','residual','ema'):raise ValueError('Invalid amplitude mode')
        if storage_dtype not in (torch.float32,torch.float16) or not math.isfinite(tau) or tau<=0:raise ValueError('Invalid state precision/tau')
        scale=torch.as_tensor(scale,dtype=torch.float32).reshape(1,-1,1,1)
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):raise ValueError('Invalid scale')
        self.register_buffer('scale',scale.clone());self.bounds=bounds;self.mode=mode;self.storage_dtype=storage_dtype;self.tau=tau
        self.last_counts=None

    def step(self,a,state,meta,reset=False,detach_state=True):
        if torch.is_grad_enabled():raise ValueError('Frozen inference only; floor gradient not specified')
        if a.dtype!=torch.float32 or a.shape[0]!=1 or a.shape[1]!=self.scale.shape[1]:raise ValueError('Single FP32 stream required')
        if not bool(torch.isfinite(a).all()) or not bool((a>=0).all()):raise ValueError('Expected finite nonnegative activation')
        self.last_counts=None
        if self.mode=='ann':return a,None,dict(state_bytes=0,reset=None,warp=False)
        q=self.scale/8.;reason='initial' if state is None else None;dt=0.
        if state is not None:
            if state['mode']!=self.mode or state['value'].dtype!=self.storage_dtype:raise ValueError('Incompatible amplitude state')
            if state['scene']!=meta['segment']:reason='scene'
            else:
                dt=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
                if dt<=0:raise ValueError('Noncausal amplitude state')
                if dt>.25:reason='gap'
        if reset:reason='explicit'
        if reason is None:past,valid=warp_bev(state['value'].float(),state['pose'],meta['pose'],self.bounds)
        else:past=torch.zeros_like(a);valid=torch.zeros_like(a[:,:1],dtype=torch.bool)
        alpha=math.exp(-dt/self.tau) if reason is None else 0.
        if self.mode=='ema':
            beta=alpha*valid.to(a.dtype);u=(1.-beta)*a+beta*past;output=u
        else:
            if self.mode=='count':alpha=0.
            v=a/q+(alpha*past if self.mode=='residual' else 0.)
            count=torch.floor(v)
            if not bool(torch.isfinite(count).all()) or float(count.max())>=2**24:raise RuntimeError('Count overflow; no clipping')
            self.last_counts=count.to(torch.int64);u=v-count;output=q*count
            if not bool(((u>=0)&(u<1)).all()):raise RuntimeError('Residual out of range')
        if not bool(torch.isfinite(output).all()):raise RuntimeError('Nonfinite amplitude output')
        error=(output-a)/q
        info=dict(state_bytes=0,reset=reason,warp=reason is None,delta_t=dt,alpha=alpha,
            valid_history_fraction=float(valid.float().mean()),normalized_error_min=float(error.min()),normalized_error_max=float(error.max()))
        if self.mode in ('count','residual'):
            # FP32 division/subtraction/multiplication allowance scales with count.
            allowance=32*torch.finfo(torch.float32).eps*(1+a/q)
            if not bool(((error>=-1-allowance)&(error<=alpha+allowance)).all()):raise RuntimeError('Amplitude error bound violated')
            info.update(error_bound_checked=True,normalized_roundoff_allowance_max=float(allowance.max()))
        if self.mode=='count':return output,None,dict(info,reset=None,warp=False)
        stored=u.to(self.storage_dtype)
        if not bool(torch.isfinite(stored).all()):raise RuntimeError('Persistent state overflow; no clipping')
        if self.mode=='residual' and not bool(((stored>=0)&(stored<=1)).all()):raise RuntimeError('Stored residual out of range')
        delta=stored.float()-u
        info.update(state_bytes=stored.numel()*stored.element_size(),state_min=float(stored.min()),state_max=float(stored.max()),
            storage_quantization_abs_max=float(delta.abs().max()),storage_quantization_mse=float(delta.square().mean()),
            decode_fp32_tensor_bytes=u.numel()*4 if reason is None and self.storage_dtype==torch.float16 else 0,
            storage_cast_logical_bytes=u.numel()*6*(1+int(reason is None)) if self.storage_dtype==torch.float16 else 0)
        new=dict(value=stored.detach(),scene=meta['segment'],pose=copy.deepcopy(meta['pose']),timestamp_us=int(meta['timestamp_us']),mode=self.mode)
        return output,new,info


class ReadoutAmplitudeAnchor(ReadoutSpikeAnchor):
    differentiable_state=False

    def __init__(self,base,mode,scale,storage_dtype=torch.float32,reset_memory=False):
        super().__init__(base,'ann',scale,storage_dtype,reset_memory)
        self.activation_memory=AmplitudeMemory(mode,scale,self.bounds,storage_dtype)
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

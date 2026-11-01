"""Frozen activation coding only: count decoding precedes dense downstream Conv.

Q16 current/membrane permits exact count versus explicit IF parity, including
threshold boundaries. Membrane resets at every real activation call, never
persists across LiDAR frames. This module does not implement sparse synapses.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector


class FixedPointIFActivation(nn.Module):
    quantum=65536

    def __init__(self, scales, mode='count', steps=8):
        super().__init__()
        if mode not in ('zero','clip','count','if') or type(steps) is not int or not 1<=steps<=1024:
            raise ValueError('Invalid activation coding mode/steps')
        scales=torch.as_tensor(scales,dtype=torch.float32).flatten()
        if not len(scales) or not bool(torch.isfinite(scales).all()) or not bool((scales>0).all()):
            raise ValueError('Positive finite per-channel scales required')
        self.register_buffer('scales',scales.reshape(1,-1,1,1).clone())
        self.mode,self.steps=mode,steps;self.calls=0;self.last_stats=None

    def forward(self, x):
        if self.training or torch.is_grad_enabled() or x.ndim!=4 or x.shape[1]!=self.scales.shape[1]:
            raise ValueError('Frozen NCHW activation coding required')
        if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite activation')
        positive=x.clamp_min(0);clipped=torch.minimum(positive,self.scales)
        self.calls+=1
        stats=dict(mode=self.mode,steps=self.steps,elements=x.numel(),
            clipped_fraction=float((positive>self.scales).float().mean()),scale_buffer_bytes=self.scales.numel()*self.scales.element_size(),
            persistent_membrane_bytes=0,if_neuron_updates=0,spike_count=None,emission_rate=None)
        if self.mode=='zero':result=torch.zeros_like(positive)
        elif self.mode=='clip':result=clipped
        else:
            drive=torch.floor((clipped/self.scales)*self.quantum).to(torch.int32).clamp_(0,self.quantum)
            if self.mode=='count':counts=torch.floor_divide(drive*self.steps,self.quantum)
            else:
                membrane=torch.zeros_like(drive);counts=torch.zeros_like(drive)
                for _ in range(self.steps):
                    membrane.add_(drive);spikes=membrane>=self.quantum
                    emitted=spikes.to(torch.int32);membrane.sub_(emitted*self.quantum);counts.add_(emitted)
                stats['if_neuron_updates']=self.steps*x.numel()
            stats.update(spike_count=int(counts.sum()),emission_rate=float(counts.sum())/(self.steps*x.numel()),
                         decoded_nonzero_fraction=float((counts>0).float().mean()),
                         saturated_count_fraction=float((counts==self.steps).float().mean()))
            result=counts.to(x.dtype)*(self.scales/self.steps)
        self.last_stats=stats
        return result


def coded_anchor(base, scales, mode='count', steps=8):
    model=AnchoredLatePyramidDetector(base,period=2,max_age=.25,coverage='strict')
    # Share frozen spatial path and heads, isolate the modified state updater.
    core=nn.Module()
    for name,module in base.core.named_children():
        core.add_module(name,copy.deepcopy(module) if name=='deep_update' else module)
    if not hasattr(core,'dir_head'):core.dir_head=None
    device=base.core.deep_update.history.weight.device
    core.deep_update.residual[1]=FixedPointIFActivation(scales,mode,steps).to(device)
    model.core=core;model.train(False)
    return model


class CodedReadout(nn.Module):
    """Two decoded input codes; BN and final activation remain analog."""
    def __init__(self, source, scales, mode='count', steps=8):
        super().__init__()
        if mode not in ('clip','count','if','zero_mid'):
            raise ValueError('Unsupported readout coding mode')
        if len(source.layers)!=1 or not isinstance(source.layers[0].double_conv[2],nn.ReLU):
            raise ValueError('Expected frozen two-convolution BN readout')
        self.body=copy.deepcopy(source)
        self.input_codec=nn.Identity() if mode=='zero_mid' else FixedPointIFActivation(scales['input'],mode,steps)
        self.body.layers[0].double_conv[2]=FixedPointIFActivation(scales['middle'],'zero' if mode=='zero_mid' else mode,steps)

    @property
    def middle_codec(self):
        return self.body.layers[0].double_conv[2]

    def forward(self,x):
        if bool((x<0).any()):raise ValueError('Readout input code requires nonnegative deblock output')
        return self.body(self.input_codec(x))

    def coding_stats(self):
        return dict(input=getattr(self.input_codec,'last_stats',None),middle=self.middle_codec.last_stats)


def readout_coded_anchor(base,scales,mode='count',steps=8):
    model=AnchoredLatePyramidDetector(base,period=2,max_age=.25,coverage='strict')
    core=nn.Module()
    for name,module in base.core.named_children():
        core.add_module(name,CodedReadout(module,scales,mode,steps) if name=='readout' else module)
    if not hasattr(core,'dir_head'):core.dir_head=None
    model.core=core.to(base.core.deep_update.history.weight.device);model.train(False)
    return model

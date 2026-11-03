"""Exact-forward event costs with explicit surrogate derivatives.

This is a training/diagnostic observer, not sparse execution or task memory.
Previous integer sending references are detached and never pose-warped.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_causal_readout_qat import CausalReadoutQAT


def event_objectives(inputs,previous,fanout,output_channels,refresh):
    if inputs.ndim!=4 or inputs.shape[0]!=1 or fanout.shape!=(1,1,*inputs.shape[-2:]):
        raise ValueError('Single stream and matching fanout required')
    if refresh:
        zero=inputs.sum()*0
        return zero,zero
    if previous is None or previous.shape!=inputs.shape or previous.dtype not in (torch.int32,torch.int64):
        raise ValueError('Matching detached integer sending reference required')
    delta=inputs-previous.detach().to(inputs.dtype)
    absolute=delta.abs();hard=(absolute!=0).to(inputs.dtype)
    soft=-torch.expm1(-absolute)
    weighted=(hard+(soft-soft.detach()))*fanout*output_channels
    units=absolute*fanout*output_channels
    return weighted.sum(),units.sum()


class EventObjectiveReadout(CausalReadoutQAT):
    def __init__(self,source,quanta):
        super().__init__(source,quanta,True)
        self.event_previous=None;self.event_next=[];self.event_records=[]
        self.event_weighted=None;self.event_units=None;self.event_refresh=None

    def begin_frame(self,refresh):
        if self.event_refresh is not None:raise ValueError('Unfinished objective frame')
        self.event_refresh=bool(refresh)
        self.event_next=[];self.event_records=[];self.event_weighted=None;self.event_units=None

    def observe_integer_inputs(self,index,layer,inputs):
        if self.event_refresh is None:raise ValueError('begin_frame must supply actual refresh decision')
        h,w=inputs.shape[-2:];kernel=layer.kernel_size[0]
        fanout=F.conv2d(inputs.new_ones(1,1,h,w),inputs.new_ones(1,1,kernel,kernel),padding=kernel//2)
        previous=None if self.event_previous is None else self.event_previous[index]
        weighted,units=event_objectives(inputs,previous,fanout,layer.out_channels,self.event_refresh)
        self.event_weighted=weighted if self.event_weighted is None else self.event_weighted+weighted
        self.event_units=units if self.event_units is None else self.event_units+units
        self.event_records.append(dict(weighted_event_mac=int(weighted.detach()),unit_spike_ac=int(units.detach()),refresh=self.event_refresh))
        self.event_next.append(inputs.detach().to(torch.int32))

    def forward(self,x):
        result=super().forward(x)
        expected=sum(isinstance(m,nn.Conv2d) for m in self.layers)
        if len(self.event_next)!=expected:raise RuntimeError('Missing input observer')
        self.event_previous=self.event_next;self.event_next=[];self.event_refresh=None
        return result

    def reference_bytes(self):
        return sum(x.numel()*x.element_size() for x in (self.event_previous or []))

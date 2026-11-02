"""Causal readout change encoding with a dense GPU backend and explicit costs.

Sending references are execution state on a fixed tensor grid, never pose-warped.
The backend recomputes Conv(q); AC/event-MAC figures describe a hypothetical
persistent synaptic accumulator, not operations skipped by this GPU module.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector, reset_reason
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv, DoubleConv


def readout_leaves(module):
    if not isinstance(module,(nn.Sequential,nn.ModuleList,DownsampleConv,DoubleConv)):raise ValueError('Unsupported readout container')
    leaves=[]
    for child in module.children():
        if list(child.children()):leaves.extend(readout_leaves(child))
        elif isinstance(child,(nn.Conv2d,nn.BatchNorm2d,nn.ReLU)):
            leaves.append(copy.deepcopy(child))
        else:raise ValueError('Only a sequential Conv/BN/ReLU readout is supported')
    return leaves


class ChangeEncodedReadout(nn.Module):
    def __init__(self,source,quanta):
        super().__init__();self.layers=nn.ModuleList(readout_leaves(source));convs=[m for m in self.layers if isinstance(m,nn.Conv2d)]
        if len(convs)!=len(quanta) or not convs:raise ValueError('One channel quantum per convolution input required')
        for i,(conv,quantum) in enumerate(zip(convs,quanta)):
            if conv.groups!=1 or conv.stride!=(1,1) or conv.dilation!=(1,1) or conv.padding_mode!='zeros' or conv.kernel_size not in ((1,1),(3,3)) or conv.padding!=(conv.kernel_size[0]//2,)*2:raise ValueError('Same-grid ungrouped 1x1 or 3x3 required')
            q=torch.as_tensor(quantum,device=conv.weight.device,dtype=conv.weight.dtype).reshape(1,-1,1,1)
            if q.numel()!=conv.in_channels or not bool(torch.isfinite(q).all()) or not bool((q>0).all()):raise ValueError('Positive finite channel quanta required')
            self.register_buffer('quantum_%d'%i,q)
        for m in self.layers:
            if isinstance(m,nn.ReLU):m.inplace=False
        self.previous=None;self.refresh=True;self.next_state=None;self.statistics=None;self.eval()

    def forward(self,x):
        if torch.is_grad_enabled() or self.training:raise ValueError('Frozen inference only')
        if x.shape[0]!=1:raise ValueError('Single stream required')
        states=[];records=[];index=0;nonlinear_elements=0
        for layer in self.layers:
            if not isinstance(layer,nn.Conv2d):
                x=layer(x);nonlinear_elements+=x.numel();continue
            quantum=getattr(self,'quantum_%d'%index);h,w=x.shape[-2:];kernel=layer.kernel_size[0]
            dense_mac=h*w*layer.in_channels*layer.out_channels*kernel*kernel
            if self.refresh or self.previous is None:
                q=x.clone();counts=torch.zeros_like(x);error=0.
            else:
                old=self.previous[index]
                if old.shape!=x.shape:raise ValueError('Shape changed without refresh')
                counts=torch.trunc((x-old)/quantum);q=old+quantum*counts
                error=float(((x-q).abs()/quantum).max());assert error<=1.001
            fanout=F.conv2d(x.new_ones(1,1,h,w),x.new_ones(1,1,kernel,kernel),padding=kernel//2)
            nonzero=counts!=0;spatial=nonzero.any(1,keepdim=True)
            active_output=F.max_pool2d(spatial.to(x.dtype),kernel,1,kernel//2)>0
            events=int(nonzero.sum());units=int(counts.abs().sum(dtype=torch.float64))
            weighted=int((nonzero.to(x.dtype)*fanout).sum(dtype=torch.float64))*layer.out_channels
            ac=int((counts.abs()*fanout).sum(dtype=torch.float64))*layer.out_channels
            records.append(dict(input_channels=layer.in_channels,output_channels=layer.out_channels,height=h,width=w,kernel=kernel,
                dense_mac=dense_mac,refresh_dense_mac=dense_mac if self.refresh or self.previous is None else 0,
                weighted_event_mac=weighted,unit_spike_ac=ac,nonzero_events=events,unit_spikes=units,
                active_input_positions=int(spatial.sum()),active_output_positions=int(active_output.sum()),
                input_elements=x.numel(),output_elements=h*w*layer.out_channels,quantization_error_in_quanta=error))
            states.append(q.detach());x=layer(q);index+=1
        self.next_state=states;self.statistics=dict(layers=records,nonlinear_elements=nonlinear_elements,
            sending_reference_bytes=sum(q.numel()*q.element_size() for q in states),
            quantum_bytes=sum(getattr(self,'quantum_%d'%i).numel()*4 for i in range(index)),
            hypothetical_accumulator_bytes=sum(r['output_elements']*4 for r in records),backend='dense_conv_of_change_encoded_input')
        return x


class ReadoutEventDetector(ProtectedMultiscaleRefreshDetector):
    def __init__(self,source,quanta):
        nn.Module.__init__(self)
        if type(source) is not ProtectedMultiscaleRefreshDetector:raise ValueError('Protected source required')
        self.core=copy.deepcopy(source.core);self.pillar_vfe,self.scatter=source.pillar_vfe,source.scatter
        for key in ('bounds','period','max_age','light_observations','memory_policy','spatial_only'):setattr(self,key,copy.deepcopy(getattr(source,key)))
        self.variant='protected_readout_change_encoded';self.core.readout=ChangeEncodedReadout(self.core.readout,quanta);self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if requests is not None or not detach_state:raise ValueError('Frozen whole-stream execution only')
        reason,_=reset_reason(state,meta);refresh=bool(reason)
        if not refresh:refresh=state['since_refresh']+1>=self.period or (int(meta['timestamp_us'])-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6
        readout=self.core.readout;readout.previous=None if state is None else state['readout'];readout.refresh=refresh
        pred,new,info=super().step_spatial(spatial,state,meta,detach_state=True)
        new['readout']=readout.next_state;info['readout_events']=readout.statistics
        info['cache_bytes']+=readout.statistics['sending_reference_bytes'];readout.previous=None;readout.next_state=None
        return pred,new,info

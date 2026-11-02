"""Integer-grid readout with matched dense and sparse causal execution.

Refresh quantization changes the ANN. Dense integer Conv is certified FP64
simulation, not an efficient deployed integer kernel.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_readout_events import ChangeEncodedReadout,ReadoutEventDetector
from opencood.models.sub_modules.int_integer_event_reference import folded_integer_weights,integer_input,dense_integer_reference,sparse_integer_delta


class IntegerReadout(ChangeEncodedReadout):
    def __init__(self,source,quanta,sparse):
        super().__init__(source,quanta);self.sparse=bool(sparse);self.previous_accumulators=None;self.next_accumulators=None
        for i,conv in enumerate(m for m in self.layers if isinstance(m,nn.Conv2d)):
            weight,scale=folded_integer_weights(conv,getattr(self,'quantum_%d'%i))
            self.register_buffer('integer_weight_%d'%i,weight);self.register_buffer('output_scale_%d'%i,scale)

    def forward(self,x):
        if torch.is_grad_enabled() or self.training:raise ValueError('Frozen integer inference only')
        if x.shape[0]!=1:raise ValueError('Single stream required')
        states=[];accumulators=[];records=[];i=0;nonlinear=0
        for layer in self.layers:
            if not isinstance(layer,nn.Conv2d):x=layer(x);nonlinear+=x.numel();continue
            quantum=getattr(self,'quantum_%d'%i);weight=getattr(self,'integer_weight_%d'%i);scale=getattr(self,'output_scale_%d'%i)
            inputs=integer_input(x,quantum);h,w=x.shape[-2:];kernel=layer.kernel_size[0];dense_mac=h*w*layer.in_channels*layer.out_channels*kernel*kernel
            refresh=self.refresh or self.previous is None
            delta=torch.zeros_like(inputs) if refresh else inputs-self.previous[i]
            if self.sparse and not refresh:
                if self.previous_accumulators is None:raise ValueError('Missing integer accumulator state')
                update,execution=sparse_integer_delta(layer,delta,weight);acc=self.previous_accumulators[i]+update
                reference_bound=0;fp64_flops=0;integer_macs=execution['executed_macs'];reference_calls=0
            else:
                acc,reference_bound=dense_integer_reference(layer,inputs,weight);execution={};fp64_flops=2*dense_mac;integer_macs=0;reference_calls=1
            # The previous and update absolute bounds are each below2**53;
            # their int64 addition cannot overflow even before cancellation.
            if bool((acc.abs()>=2**54).any()):raise ValueError('Integer accumulator state bound exceeded')
            fanout=F.conv2d(x.new_ones(1,1,h,w),x.new_ones(1,1,kernel,kernel),padding=kernel//2).to(torch.int64)
            active=delta!=0;weighted=int((active.to(torch.int64)*fanout).sum())*layer.out_channels;units=int((delta.abs()*fanout).sum())*layer.out_channels
            records.append(dict(input_channels=layer.in_channels,output_channels=layer.out_channels,height=h,width=w,kernel=kernel,dense_mac=dense_mac,refresh=refresh,
                refresh_dense_mac=dense_mac if refresh else 0,weighted_event_mac=weighted,unit_spike_ac=units,nonzero_events=int(active.sum()),unit_spikes=int(delta.abs().sum()),
                actual_fp64_conv_flops=fp64_flops,actual_integer_macs=integer_macs,dense_reference_calls=reference_calls,reference_absolute_bound=reference_bound,
                event_absolute_bound=execution.get('absolute_accumulation_bound',0),integer_weight_bytes=weight.numel()*weight.element_size(),scale_bytes=scale.numel()*scale.element_size(),
                max_input_integer=int(inputs.abs().max()),quantization_error_in_quanta=float((x.double()/quantum.double()-inputs.double()).abs().max())))
            if self.sparse and not refresh:assert integer_macs==weighted
            states.append(inputs.detach());accumulators.append(acc.detach());x=(acc.double()*scale.reshape(1,-1,1,1)).to(x.dtype)
            if layer.bias is not None:x=x+layer.bias.reshape(1,-1,1,1)
            i+=1
        self.next_state=states;self.next_accumulators=accumulators
        self.statistics=dict(layers=records,nonlinear_elements=nonlinear,sending_reference_bytes=sum(q.numel()*q.element_size() for q in states),accumulator_bytes=sum(q.numel()*q.element_size() for q in accumulators),
            quantum_bytes=sum(getattr(self,'quantum_%d'%j).numel()*4 for j in range(i)),derived_weight_bytes=sum(r['integer_weight_bytes'] for r in records),derived_scale_bytes=sum(r['scale_bytes'] for r in records),backend='sparse_int64_events' if self.sparse else 'bounded_fp64_integer_reference')
        return x


class IntegerReadoutDetector(ReadoutEventDetector):
    def __init__(self,source,quanta,sparse=False):
        super().__init__(source,quanta);self.core.readout=IntegerReadout(source.core.readout,quanta,sparse);self.variant='protected_integer_readout_sparse' if sparse else 'protected_integer_readout_dense';self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        readout=self.core.readout;readout.previous_accumulators=None if state is None else state['readout_accumulators']
        pred,new,info=super().step_spatial(spatial,state,meta,detach_state,requests)
        new['readout_accumulators']=readout.next_accumulators;info['cache_bytes']+=readout.statistics['accumulator_bytes']
        readout.previous_accumulators=None;readout.next_accumulators=None;return pred,new,info

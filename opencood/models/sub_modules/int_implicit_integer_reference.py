"""Retain only external sending reference; derive internal ones from old A."""
import torch
from torch import nn
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout,IntegerReadoutDetector
from opencood.models.sub_modules.int_integer_event_reference import integer_input
from opencood.models.sub_modules.int_compact_causal_integer_state import narrow_integer_state


class ImplicitReferenceIntegerReadout(IntegerReadout):
    def reconstruct_references(self,external,accumulators,dtype=torch.float32):
        if torch.is_grad_enabled() or self.training:raise ValueError('Frozen reconstruction only')
        count=sum(isinstance(m,nn.Conv2d) for m in self.layers)
        if len(external)!=1 or len(accumulators)!=count or dtype!=torch.float32:raise ValueError('One external reference, all accumulators and FP32 decode required')
        references=[external[0]];old=None;i=0
        for layer in self.layers:
            if isinstance(layer,nn.Conv2d):
                if i:
                    references.append(integer_input(old,getattr(self,'quantum_%d'%i)))
                if i==count-1:break
                old=(accumulators[i].double()*getattr(self,'output_scale_%d'%i).reshape(1,-1,1,1)).to(dtype)
                if layer.bias is not None:old=old+layer.bias.reshape(1,-1,1,1)
                i+=1
            elif old is not None:old=layer(old)
        if len(references)!=count:raise RuntimeError('Reference reconstruction incomplete')
        return references

    def forward(self,x):
        if x.dtype!=torch.float32:raise ValueError('Fixed FP32 output decode required')
        original=self.previous;reconstructed=0
        if original is not None and not self.refresh:
            self.previous=self.reconstruct_references(original,self.previous_accumulators,x.dtype)
            reconstructed=sum(v.numel() for v in self.previous[1:])
        try:output=super().forward(x)
        finally:self.previous=original
        # Internal n is fully determined by old A and fixed layer transforms.
        # Never retain reconstructed references or discarded next references.
        external=self.next_state[0];self.next_state=[narrow_integer_state(external,True)]
        self.next_accumulators=[narrow_integer_state(v,False) for v in self.next_accumulators]
        self.statistics.update(sending_reference_bytes=sum(v.numel()*v.element_size() for v in self.next_state),
            accumulator_bytes=sum(v.numel()*v.element_size() for v in self.next_accumulators),
            reference_dtypes=[str(v.dtype) for v in self.next_state],accumulator_dtypes=[str(v.dtype) for v in self.next_accumulators],
            reconstructed_reference_elements=reconstructed,stored_reference_count=1,
            storage_policy='Only external n plus all compact A persist; internal n reconstructed from previous A through identical FP32 decode/BN/ReLU/quantization. Reconstruction is transient and adds non-Conv work; no latency claim.')
        return output


class ImplicitReferenceIntegerDetector(IntegerReadoutDetector):
    def __init__(self,source,quanta,sparse=True):
        super().__init__(source,quanta,sparse)
        self.core.readout=ImplicitReferenceIntegerReadout(source.core.readout,quanta,sparse)
        self.variant='implicit_reference_integer_sparse' if sparse else 'implicit_reference_integer_dense'
        self.eval()

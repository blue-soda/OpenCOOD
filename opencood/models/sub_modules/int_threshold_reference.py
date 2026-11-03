"""Frozen monotone integer thresholds for internal sending-reference decode.

This replaces old-reference float decode only. Current readout, task memory,
and all convolution work are unchanged. Tables are derived static memory.
"""
import math
import torch
from torch import nn
from opencood.models.sub_modules.int_integer_event_reference import integer_input
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout,ImplicitReferenceIntegerDetector


class IntegerThresholdMap:
    def __init__(self, oracle, orientation, hw, levels=4096, bound=2**31-2):
        if torch.is_grad_enabled() or levels<1 or bound<1 or bound>=2**31-1:raise ValueError('Frozen bounded threshold construction required')
        self.oracle=oracle;self.hw=tuple(hw);self.levels=levels;self.bound=bound
        self.orientation=orientation.detach().reshape(-1,1).to(torch.int8)
        if not bool(((self.orientation==1)|(self.orientation==-1)).all()):raise ValueError('Orientation must be signed unit')
        channels=len(self.orientation);positions=hw[0]*hw[1];width=2*levels+2
        if width>positions:raise ValueError('Calibration grid must hold all signed thresholds')
        targets=torch.arange(-levels,levels+2,device=orientation.device,dtype=torch.int64).reshape(1,-1).expand(channels,-1)
        lo=torch.full_like(targets,-bound-1);hi=torch.full_like(targets,bound+1);self.build_oracle_calls=0
        def evaluate(values):
            padded=torch.zeros(channels,positions,dtype=torch.int64,device=orientation.device)
            padded[:,:width]=values*self.orientation.to(torch.int64)
            self.build_oracle_calls+=1
            return oracle(padded.reshape(1,channels,*hw)).reshape(channels,-1)[:,:width]
        while bool(((hi-lo)>1).any()):
            active=(hi-lo)>1;mid=torch.div(lo+hi,2,rounding_mode='floor');value=evaluate(mid.clamp(-bound,bound));upper=value>=targets
            hi=torch.where(active&upper,mid,hi);lo=torch.where(active&~upper,mid,lo)
        values=evaluate(hi.clamp(-bound,bound));lower=evaluate((hi-1).clamp(-bound,bound))
        if not bool(((hi>bound)|(values>=targets)).all()) or not bool(((hi<=-bound)|(lower<targets)).all()):raise RuntimeError('Threshold boundary certification failed')
        if not bool((hi[:,1:]>=hi[:,:-1]).all()):raise RuntimeError('Nonmonotone thresholds')
        self.thresholds=hi.to(torch.int32).contiguous()
        self.table_bytes=self.thresholds.numel()*self.thresholds.element_size()+self.orientation.numel()*self.orientation.element_size()
        self.build_float_elements=self.build_oracle_calls*channels*positions

    def decode(self,accumulator):
        if torch.is_grad_enabled() or tuple(accumulator.shape)!=(1,len(self.orientation),*self.hw):raise ValueError('Frozen calibration grid required')
        oriented=accumulator.to(torch.int64).reshape(len(self.orientation),-1)*self.orientation.to(torch.int64)
        outside=(oriented < -self.bound)|(oriented>self.bound)
        index=torch.searchsorted(self.thresholds,oriented.clamp(-self.bound,self.bound).to(torch.int32).contiguous(),right=True)
        invalid=outside|(index==0)|(index==self.thresholds.shape[1]);fallback=int(invalid.sum())
        # A full-layer oracle fallback preserves exact values outside coverage.
        value=self.oracle(accumulator) if fallback else (index-self.levels-1).reshape_as(accumulator)
        return value,dict(lookup_elements=accumulator.numel(),comparison_upper_bound=accumulator.numel()*math.ceil(math.log2(self.thresholds.shape[1]+1)),
                          fallback_elements=fallback,float_fallback_elements=accumulator.numel() if fallback else 0)


class ThresholdImplicitReferenceReadout(ImplicitReferenceIntegerReadout):
    def __init__(self,source,quanta,sparse=True):
        super().__init__(source,quanta,sparse);self.threshold_maps=None;self.threshold_step=[];self.threshold_build_elements=0
        self.capture_reference_audit=False;self.reference_audit=None

    def reconstruct_references(self,external,accumulators,dtype=torch.float32):
        if torch.is_grad_enabled() or self.training or dtype!=torch.float32 or len(external)!=1:raise ValueError('Frozen FP32 source mapping required')
        if self.threshold_maps is None:
            groups=[]
            for layer in self.layers:
                if isinstance(layer,nn.Conv2d):groups.append([layer,[]])
                else:groups[-1][1].append(layer)
            if len(accumulators)!=len(groups):raise ValueError('Accumulator count mismatch')
            self.threshold_maps=[]
            for i,(conv,tail) in enumerate(groups[:-1]):
                scale=getattr(self,'output_scale_%d'%i);quantum=getattr(self,'quantum_%d'%(i+1))
                orientation=torch.ones(conv.out_channels,dtype=torch.int8,device=scale.device)
                for layer in tail:
                    if isinstance(layer,nn.BatchNorm2d):orientation*=torch.where(layer.weight<0,-1,1).to(torch.int8)
                    elif not isinstance(layer,nn.ReLU):raise ValueError('Only affine BN and ReLU supported')
                def oracle(acc,conv=conv,tail=tail,scale=scale,quantum=quantum):
                    value=(acc.double()*scale.reshape(1,-1,1,1)).float()
                    if conv.bias is not None:value=value+conv.bias.reshape(1,-1,1,1)
                    for layer in tail:value=layer(value)
                    return integer_input(value,quantum)
                decoder=IntegerThresholdMap(oracle,orientation,accumulators[i].shape[-2:]);self.threshold_maps.append(decoder)
                self.threshold_build_elements+=decoder.build_float_elements
        references=[external[0]]
        for decoder,acc in zip(self.threshold_maps,accumulators[:-1]):
            value,stats=decoder.decode(acc);references.append(value);self.threshold_step.append(stats)
        if self.capture_reference_audit:self.reference_audit=references[1:]
        return references

    def forward(self,x):
        self.threshold_step=[];self.threshold_build_elements=0;self.reference_audit=None
        result=super().forward(x)
        self.statistics.update(threshold_table_bytes=sum(m.table_bytes for m in self.threshold_maps or []),
            threshold_lookup_elements=sum(z['lookup_elements'] for z in self.threshold_step),
            threshold_comparison_upper_bound=sum(z['comparison_upper_bound'] for z in self.threshold_step),
            threshold_fallback_elements=sum(z['fallback_elements'] for z in self.threshold_step),
            threshold_float_fallback_elements=sum(z['float_fallback_elements'] for z in self.threshold_step),
            threshold_build_float_elements=self.threshold_build_elements,
            threshold_reference_audit_bytes=sum(v.numel()*v.element_size() for v in self.reference_audit or []),
            reference_decoder='Integer monotone lookup for old internal n; fixed geometry, int32 tables, full float layer fallback on out-of-range values. Current decode remains float; tables are static overhead, no task-memory benefit or speed claim.')
        return result


class ThresholdImplicitReferenceDetector(ImplicitReferenceIntegerDetector):
    def __init__(self,source,quanta):
        super().__init__(source,quanta,True);self.core.readout=ThresholdImplicitReferenceReadout(source.core.readout,quanta,True)
        self.variant='threshold_implicit_reference_integer';self.eval()

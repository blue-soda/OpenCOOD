"""Bounded integer-dot reference using dense FP32 GEMM, then INT32 block sums.

Not a native INT8 convolution kernel. TF32 must be disabled; outputs are checked
against integer/bound invariants and sampled independent INT64 products.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_folded_readout import FoldedCodeConv


def bitplane_cost(counts,out_channels):
    """Ideal valid spatial deliveries, no skipping of zero weights; not GPU work."""
    assert counts.dtype==torch.int32 and bool(((counts>=0)&(counts<=8)).all())
    h,w=counts.shape[-2:]
    y=torch.arange(h,device=counts.device);x=torch.arange(w,device=counts.device)
    fy=sum(((y+d>=0)&(y+d<h)).long() for d in (-1,0,1))
    fx=sum(((x+d>=0)&(x+d<w)).long() for d in (-1,0,1))
    fanout=fy[:,None]*fx[None,:]*out_channels
    histogram=[int((counts==v).sum()) for v in range(9)]
    weighted=[int(((counts==v).long()*fanout).sum()) for v in range(9)]
    pop=[bin(v).count('1') for v in range(9)]
    return dict(count_histogram=histogram,valid_delivery_histogram=weighted,
        unary_event_ac=sum(v*n for v,n in enumerate(weighted)),
        sparse_count_mac=sum(weighted[1:]),bitplane_shifted_ac=sum(p*n for p,n in zip(pop,weighted)),
        bitplane_source_packets=sum(p*n for p,n in zip(pop,histogram)),
        unary_source_packets=sum(v*n for v,n in enumerate(histogram)),
        count_source_packets=sum(histogram[1:]),
        scope='Ideal nonzero activations, valid 3x3 fanout, all output channels, no zero-weight skipping; shift/routing/memory/neuron costs separate')


def bounded_integer_conv(values,qweight,max_input=8,audit_seed=None):
    assert values.dtype==torch.int32 and qweight.dtype==torch.int8
    assert qweight.shape[1]==values.shape[1] and qweight.shape[2:]==(3,3)
    assert bool((values>=0).all()) and bool((values<=max_input).all())
    assert bool((qweight>=-127).all())
    bound=values.shape[1]*9*127*max_input
    assert bound<2**24 and bound<2**31
    if values.is_cuda:assert not torch.backends.cuda.matmul.allow_tf32
    columns=F.unfold(values.float(),3,padding=1)
    weights=qweight.reshape(qweight.shape[0],-1).float()
    raw=torch.matmul(weights,columns)
    assert torch.equal(raw,torch.round(raw)) and bool((raw.abs()<=bound).all())
    result=raw.to(torch.int32)
    checks=0
    if audit_seed is not None:
        channel=(audit_seed*37)%qweight.shape[0]
        positions=sorted({0,columns.shape[-1]//2,columns.shape[-1]-1})
        expected=torch.stack([(columns[0,:,p].long()*weights[channel].long()).sum() for p in positions])
        assert torch.equal(result[0,channel,positions].long(),expected)
        checks=len(positions)
    return result.reshape(values.shape[0],qweight.shape[0],values.shape[-2],values.shape[-1]),checks


class IntegerCodeConv(FoldedCodeConv):
    def __init__(self,source,scales,mode):
        super().__init__(source,scales,'folded_count' if mode=='bitplane' else mode,8)
        self.mode=mode
        self.calls=0

    def forward(self,x):
        if self.training or torch.is_grad_enabled():raise ValueError('Frozen diagnostic only')
        self.calls+=1
        positive=x.clamp_min(0);overflow=positive>self.scales
        drive=torch.floor(torch.minimum(positive,self.scales)/self.scales*65536).to(torch.int32).clamp_(0,65536)
        drive.masked_fill_(overflow,0)
        counts=torch.floor_divide(drive*8+32768,65536)
        seed=self.calls if self.calls<=32 else None
        checks=0
        if self.mode=='folded_count':
            accumulator,checks=bounded_integer_conv(counts,self.event_int8,8,seed)
            calls=2;integer_adds=0
        elif self.mode=='bitplane':
            accumulator=None
            for bit in range(4):
                plane=(counts>>bit)&1
                contribution,nchecks=bounded_integer_conv(plane,self.event_int8,1,seed)
                contribution=contribution<<bit
                accumulator=contribution if accumulator is None else accumulator+contribution
                checks+=nchecks
            calls=5;integer_adds=3*accumulator.numel()
        else:
            membrane=torch.full_like(drive,32768);emitted=torch.zeros_like(drive);accumulator=None
            for _ in range(8):
                membrane.add_(drive);spikes=(membrane>=65536).to(torch.int32)
                membrane.sub_(spikes*65536);emitted.add_(spikes)
                contribution,nchecks=bounded_integer_conv(spikes,self.event_int8,1,seed)
                accumulator=contribution if accumulator is None else accumulator+contribution
                checks+=nchecks
            assert torch.equal(emitted,counts)
            calls=9;integer_adds=7*accumulator.numel()
        bound=self.source.in_channels*9*127*8
        assert accumulator.dtype==torch.int32 and bool((accumulator.abs()<=bound).all())
        output=accumulator.float()*self.event_weight_scale.reshape(1,-1,1,1)
        tail=torch.where(overflow,positive,torch.zeros_like(positive))
        output=output+F.conv2d(tail,self.tail_weight,None,1,1)
        if self.source.bias is not None:output=output+self.source.bias.reshape(1,-1,1,1)
        reference_flops=2*output.numel()*self.source.in_channels*9
        self.last_stats=dict(functional_conv_flops=calls*reference_flops,reference_conv_flops=reference_flops,
            dense_conv_calls=calls,spike_count=int(counts.sum()),analog_elements=int(overflow.sum()),
            folded_weight_bytes=self.event_weight.numel()*self.event_weight.element_size(),
            integer_dot_bound=bound,integer_time_block_additions=integer_adds,
            post_accumulator_scale_muls=output.numel(),sampled_int64_dot_checks=checks,
            accumulator_dtype='int32',dot_backend='dense unfold+FP32 GEMM with integer checks; not native INT8')
        if self.mode=='bitplane':
            self.last_stats['integer_bit_shifts']=3*accumulator.numel()
            self.last_stats['ideal_spatial_cost']=bitplane_cost(counts,self.source.out_channels)
        return output


def integer_readout(source,mode):
    result=copy.deepcopy(source);block=result.body.layers[0].double_conv
    block[0]=IntegerCodeConv(block[0],source.input_codec.scales,mode)
    block[3]=IntegerCodeConv(block[3],source.middle_codec.scales,mode)
    result.input_codec=nn.Identity();block[2]=nn.Identity()
    return result.eval()

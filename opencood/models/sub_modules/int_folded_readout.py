"""Numerical experiment: scale-folded count / time-block event convolutions.

All convolutions execute densely. Not a sparse backend or neuromorphic device
emulator: the inner convolution accumulation order is the library's choice.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FoldedCodeConv(nn.Module):
    def __init__(self, source, scales, mode):
        super().__init__()
        if mode not in ('folded_count', 'event8'):
            raise ValueError('Unsupported frozen diagnostic mode')
        assert source.kernel_size == (3, 3) and source.stride == (1, 1)
        assert source.padding == (1, 1) and source.dilation == (1, 1) and source.groups == 1
        assert source.padding_mode == 'zeros'
        self.source = copy.deepcopy(source)
        self.mode = mode
        self.register_buffer('scales', scales.detach().clone().to(source.weight))
        self.register_buffer('event_weight', (source.weight.detach()*self.scales.reshape(1,-1,1,1)/8).clone())
        self.last_stats = None

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise ValueError('Frozen inference diagnostic only')
        positive = x.clamp_min(0)
        overflow = positive > self.scales
        drive = torch.floor(torch.minimum(positive,self.scales)/self.scales*65536).to(torch.int32).clamp_(0,65536)
        drive.masked_fill_(overflow,0)
        counts = torch.floor_divide(drive*8+32768,65536)
        def conv(value, weight):
            return F.conv2d(value,weight,None,1,1)
        if self.mode == 'folded_count':
            output = conv(counts.to(x.dtype),self.event_weight)
            calls = 2
        else:
            membrane = torch.full_like(drive,32768)
            emitted_counts = torch.zeros_like(drive)
            output = None
            for _ in range(8):
                membrane.add_(drive)
                spikes = (membrane >= 65536).to(torch.int32)
                membrane.sub_(spikes*65536)
                emitted_counts.add_(spikes)
                contribution = conv(spikes.to(x.dtype),self.event_weight)
                output = contribution if output is None else output+contribution
            assert torch.equal(emitted_counts,counts)
            calls = 9
        output = output+conv(torch.where(overflow,positive,torch.zeros_like(positive)),self.source.weight)
        if self.source.bias is not None:
            output = output+self.source.bias.reshape(1,-1,1,1)
        reference_flops = 2*output.numel()*self.source.in_channels*9
        self.last_stats = dict(functional_conv_flops=calls*reference_flops,
            reference_conv_flops=reference_flops,dense_conv_calls=calls,
            spike_count=int(counts.sum()),analog_elements=int(overflow.sum()),
            folded_weight_bytes=self.event_weight.numel()*self.event_weight.element_size())
        return output


def folded_readout(source,mode):
    result = copy.deepcopy(source)
    block = result.body.layers[0].double_conv
    assert len(block) == 6 and isinstance(block[5],nn.ReLU)
    first = FoldedCodeConv(block[0],source.input_codec.scales,mode)
    second = FoldedCodeConv(block[3],source.middle_codec.scales,mode)
    result.input_codec = nn.Identity()
    block[0],block[2],block[3] = first,nn.Identity(),second
    return result.eval()


def folded_stats(readout):
    block = readout.body.layers[0].double_conv
    return dict(input=block[0].last_stats,middle=block[3].last_stats)

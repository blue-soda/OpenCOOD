"""Bounded integer event arithmetic with a certified FP64 dense reference.

This quantizes folded weights and refresh inputs; it is not the original ANN.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_channel_events import channel_event_delta


def folded_integer_weights(conv,quantum):
    folded=conv.weight.detach().double()*quantum.double().reshape(1,-1,1,1)
    scale=folded.abs().flatten(1).amax(1).clamp_min(1e-12)/32767
    weight=torch.round(folded/scale.reshape(-1,1,1,1)).to(torch.int16)
    return weight,scale


def integer_input(x,quantum):
    value=torch.round(x.double()/quantum.double().reshape(1,-1,1,1))
    if not bool(torch.isfinite(value).all()) or float(value.abs().max())>=2**31:raise ValueError('Input integer range exceeded')
    return value.to(torch.int64)


def dense_integer_reference(conv,inputs,weight):
    if inputs.dtype!=torch.int64 or weight.dtype!=torch.int16:raise ValueError('int64 inputs and int16 weights required')
    bound=int(weight.to(torch.int64).abs().flatten(1).sum(1).max())*int(inputs.abs().max())
    if bound>=2**53:raise ValueError('FP64 exact integer accumulation bound exceeded')
    output=F.conv2d(inputs.double(),weight.double(),None,conv.stride,conv.padding)
    if not torch.equal(output,output.round()):raise RuntimeError('Dense integer reference not integral')
    return output.to(torch.int64),bound


def sparse_integer_delta(conv,inputs,weight):
    if inputs.dtype!=torch.int64 or weight.dtype!=torch.int16:raise ValueError('int64 events and int16 weights required')
    bound=int(weight.to(torch.int64).abs().flatten(1).sum(1).max())*int(inputs.abs().max())
    if bound>=2**53:raise ValueError('Integer event accumulation bound exceeded')
    output,stats=channel_event_delta(conv,inputs,weight_override=weight);stats['absolute_accumulation_bound']=bound
    return output,stats

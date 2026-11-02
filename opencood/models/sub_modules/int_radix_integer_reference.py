"""Exact signed integer Conv reference via bounded FP32 radix GEMMs.

This is reference tooling, not a native integer or spiking hardware kernel.
"""
import torch
from torch.nn import functional as F


def radix_integer_reference(conv,inputs,weight,chunk=256):
    if torch.is_grad_enabled() or torch.is_autocast_enabled():raise ValueError('Frozen non-autocast reference only')
    if inputs.dtype!=torch.int64 or weight.dtype!=torch.int16:raise ValueError('int64 inputs/int16 weights required')
    if conv.groups!=1 or conv.stride!=(1,1) or conv.dilation!=(1,1) or conv.kernel_size not in ((1,1),(3,3)) or conv.padding!=(conv.kernel_size[0]//2,)*2 or inputs.shape[0]!=1:raise ValueError('Same-grid ungrouped Conv required')
    if type(chunk) is not int or not 1<=chunk<=256:raise ValueError('Exactness needs at most256 products per partial')
    if inputs.is_cuda and torch.backends.cuda.matmul.allow_tf32:raise ValueError('TF32 must be disabled')
    absolute_weight=weight.to(torch.int64).abs();absolute_input=inputs.abs();maximum_input=int(absolute_input.max());maximum_weight=int(absolute_weight.max())
    bound=int(absolute_weight.flatten(1).sum(1).max())*maximum_input
    if bound>=2**53:raise ValueError('Accumulation bound exceeded')
    height,width=inputs.shape[-2:];kernel=conv.kernel_size[0];fanin=weight.shape[1]*kernel*kernel;output=torch.zeros(conv.out_channels,height*width,device=inputs.device,dtype=torch.int64)
    input_digits=(maximum_input.bit_length()+7)//8;weight_digits=(maximum_weight.bit_length()+7)//8;macs=0;gemms=0
    signed_weight=weight.to(torch.int64).sign();signed_input=inputs.sign()
    for wi in range(weight_digits):
        wd=((absolute_weight//(1<<(8*wi)))%256*signed_weight).reshape(conv.out_channels,fanin).float()
        for xi in range(input_digits):
            xd=((absolute_input//(1<<(8*xi)))%256*signed_input).float()
            columns=F.unfold(xd,kernel,padding=kernel//2)[0] if kernel==3 else xd.reshape(fanin,height*width)
            for start in range(0,fanin,chunk):
                end=min(start+chunk,fanin);partial=torch.matmul(wd[:,start:end],columns[start:end])
                output.add_(partial.to(torch.int64),alpha=1<<(8*(wi+xi)));macs+=(end-start)*conv.out_channels*height*width;gemms+=1
    return output.reshape(1,conv.out_channels,height,width),dict(fp32_reference_macs=macs,fp32_reference_flops=2*macs,partial_gemms=gemms,chunk=chunk,partial_absolute_bound=chunk*255*255,full_absolute_bound=bound,input_digits=input_digits,weight_digits=weight_digits)

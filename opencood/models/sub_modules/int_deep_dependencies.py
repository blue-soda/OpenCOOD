"""Dependency-complete arithmetic for prospective local deep refresh.

No inference backend is implemented here. Masks trace real residual and
non-overlapping transpose-convolution dependencies, including input borders.
"""
import torch
from torch.nn import functional as F


def conv_input_mask(mask,conv,input_shape):
    if conv.groups!=1 or conv.dilation!=(1,1) or conv.padding_mode!='zeros':
        raise ValueError('Ordinary undilated zero-padded convolution required')
    base=tuple((o-1)*s-2*p+k for o,s,p,k in zip(mask.shape[-2:],conv.stride,conv.padding,conv.kernel_size))
    extra=tuple(i-b for i,b in zip(input_shape,base))
    if any(not 0<=e<s for e,s in zip(extra,conv.stride)):
        raise ValueError('Incompatible convolution output/input shapes')
    kernel=mask.new_ones((1,1,*conv.kernel_size),dtype=torch.float)
    return F.conv_transpose2d(mask.float(),kernel,stride=conv.stride,padding=conv.padding,output_padding=extra)>0


def output_shape(conv,shape):
    return tuple((i+2*p-k)//s+1 for i,p,k,s in zip(shape,conv.padding,conv.kernel_size,conv.stride))


def stage_dependencies(stage,mask,input_shape):
    shapes=[tuple(input_shape)]
    for block in stage:
        shapes.append(output_shape(block.conv2,output_shape(block.conv1,shapes[-1])))
    if mask.shape[-2:]!=shapes[-1]:raise ValueError('Stage output mask shape mismatch')
    rows=[];macs=0
    for index in reversed(range(len(stage))):
        block=stage[index];middle_shape=output_shape(block.conv1,shapes[index])
        middle=conv_input_mask(mask,block.conv2,middle_shape)
        needed=conv_input_mask(middle,block.conv1,shapes[index])
        n1,n2=int(middle.sum()),int(mask.sum())
        value=n1*block.conv1.weight.numel()+n2*block.conv2.weight.numel()
        if block.downsample is None:
            if needed.shape!=mask.shape:raise ValueError('Identity skip shape mismatch')
            needed=needed|mask;skip=0
        else:
            conv=block.downsample[0]
            needed=needed|conv_input_mask(mask,conv,shapes[index]);skip=n2*conv.weight.numel();value+=skip
        rows.append(dict(block=index,conv1_cells=n1,conv2_cells=n2,skip_macs=skip,macs=value))
        macs+=value;mask=needed
    return mask,macs,list(reversed(rows))


def deep_dependencies(second,third,deblocks,projection,mask,h_shape):
    if mask.dtype!=torch.bool or mask.shape[:2]!=(1,1) or mask.shape[-2:]!=tuple(h_shape):
        raise ValueError('Single D-grid boolean mask matching H required')
    required=[];decode_macs=0
    for deblock in deblocks:
        conv=deblock[0]
        if (not isinstance(conv,torch.nn.ConvTranspose2d) or conv.kernel_size!=conv.stride or
                conv.padding!=(0,0) or conv.output_padding!=(0,0) or conv.dilation!=(1,1) or conv.groups!=1):
            raise ValueError('Nonoverlapping zero-padding-free deconvolution required')
        required.append(F.max_pool2d(mask.float(),conv.kernel_size,conv.stride)>0)
        decode_macs+=int(mask.sum())*conv.in_channels*conv.out_channels
    if projection.kernel_size!=(1,1) or projection.stride!=(1,1) or projection.groups!=1:
        raise ValueError('Pointwise deep projection required')
    need_second,mac3,rows3=stage_dependencies(third,required[1],required[0].shape[-2:])
    second_mask=required[0]|need_second
    need_h,mac2,rows2=stage_dependencies(second,second_mask,h_shape)
    projection_macs=int(mask.sum())*projection.weight.numel()
    return dict(macs=mac2+mac3+decode_macs+projection_macs,backbone_macs=mac2+mac3,
        decode_projection_macs=decode_macs+projection_macs,required_h=need_h,
        second_output_cells=int(second_mask.sum()),third_output_cells=int(required[1].sum()),
        second_layers=rows2,third_layers=rows3)

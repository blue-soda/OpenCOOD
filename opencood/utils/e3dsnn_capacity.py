"""Function-preserving BEV channel duplication with zero-sum symmetry breaking."""
import copy
import torch
from torch import nn


def _conv(module, inputs, outputs, generator):
    transpose=isinstance(module,nn.ConvTranspose2d)
    if module.groups!=1:raise ValueError('Grouped BEV widening not supported')
    args=dict(in_channels=len(inputs),out_channels=len(outputs),kernel_size=module.kernel_size,
              stride=module.stride,padding=module.padding,dilation=module.dilation,bias=module.bias is not None)
    if transpose:args['output_padding']=module.output_padding
    widened=(nn.ConvTranspose2d if transpose else nn.Conv2d)(**args).to(module.weight)
    # Work in [output,input,kernel_y,kernel_x], including transpose convolutions.
    weight=module.weight.detach().cpu()
    if transpose:weight=weight.permute(1,0,2,3)
    counts=torch.bincount(inputs,minlength=module.in_channels)
    if torch.any(counts==0):raise ValueError('Every original input must survive widening')
    new=weight.index_select(0,outputs).index_select(1,inputs)/counts[inputs].view(1,-1,1,1)
    if torch.all(counts==2):
        positions=[torch.nonzero(inputs==i,as_tuple=False).flatten().tolist() for i in range(module.in_channels)]
        first=torch.tensor([p[0] for p in positions]);second=torch.tensor([p[1] for p in positions])
        # Pure replication leaves duplicate neurons tied under SGD. Different
        # outgoing weights with zero sum preserve the initial function while
        # allowing their gradients to diverge. Noise is not a new loss term.
        delta=torch.randn((len(outputs),module.in_channels,*module.kernel_size),generator=generator,dtype=weight.dtype)
        delta*=float(weight.std())*1e-3
        new[:,first]+=delta;new[:,second]-=delta
    elif not torch.all(counts==1):raise ValueError('Only identity or twofold input maps are supported')
    if transpose:new=new.permute(1,0,2,3)
    with torch.no_grad():
        widened.weight.copy_(new)
        if module.bias is not None:widened.bias.copy_(module.bias.detach().cpu()[outputs])
    return widened


def _bn(module, mapping):
    widened=nn.BatchNorm2d(len(mapping),eps=module.eps,momentum=module.momentum,
                         affine=module.affine,track_running_stats=module.track_running_stats)
    with torch.no_grad():
        if module.affine:
            widened.weight.copy_(module.weight.detach().cpu()[mapping]);widened.bias.copy_(module.bias.detach().cpu()[mapping])
        if module.track_running_stats:
            widened.running_mean.copy_(module.running_mean.detach().cpu()[mapping])
            widened.running_var.copy_(module.running_var.detach().cpu()[mapping])
            widened.num_batches_tracked.copy_(module.num_batches_tracked)
    return widened


def widen_backend(model, seed=20260929):
    """Expand BEV and heads in place on CPU; source features/wire stay untouched."""
    if next(model.backbone_2d.parameters()).device.type!='cpu':
        raise ValueError('Prepare widened checkpoints on CPU before moving to CUDA')
    with torch.random.fork_rng(devices=[]):
        generator=torch.Generator().manual_seed(seed)
        # Legacy AttributeConfig.__getattr__ raises KeyError for __deepcopy__.
        # Forward uses the built modules; retain the configuration as plain data.
        old_config=model.backbone_2d.model_cfg
        bev=copy.deepcopy(model.backbone_2d,{id(old_config):copy.deepcopy(dict(old_config))})
        old_outputs=[]
        for i,block in enumerate(bev.blocks):
            first=True
            for j,module in enumerate(block):
                if isinstance(module,nn.Conv2d):
                    inputs=torch.arange(module.in_channels)
                    if i>0 or not first:inputs=inputs.repeat(2)
                    block[j]=_conv(module,inputs,torch.arange(module.out_channels).repeat(2),generator);first=False
                elif isinstance(module,nn.BatchNorm2d):block[j]=_bn(module,torch.arange(module.num_features).repeat(2))
        for block in bev.deblocks:
            for j,module in enumerate(block):
                if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
                    old_outputs.append(module.out_channels)
                    block[j]=_conv(module,torch.arange(module.in_channels).repeat(2),torch.arange(module.out_channels).repeat(2),generator)
                elif isinstance(module,nn.BatchNorm2d):block[j]=_bn(module,torch.arange(module.num_features).repeat(2))
        if len(old_outputs)!=2:raise ValueError('Expected exactly two upsample branches')
        offset=0;parts=[]
        for width in old_outputs:
            parts.append(torch.arange(offset,offset+width).repeat(2));offset+=width
        # Concatenation is [branch0,copy0,branch1,copy1], not [all,copy_all].
        inputs=torch.cat(parts)
        for name in ('cls_head','reg_head','dir_head'):
            module=getattr(model,name)
            setattr(model,name,_conv(module,inputs,torch.arange(module.out_channels),generator))
        bev.num_bev_features*=2
        bev.model_cfg=copy.deepcopy(bev.model_cfg)
        for key in ('NUM_FILTERS','NUM_UPSAMPLE_FILTERS'):bev.model_cfg[key]=[2*x for x in bev.model_cfg[key]]
        model.backbone_2d=bev
    return model

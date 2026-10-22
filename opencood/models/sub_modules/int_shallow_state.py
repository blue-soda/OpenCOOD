"""Mature observation stem and a history-conditioned, local spatial update.

The identity-initialized mixer preserves the original dense encoder. This is
an architectural scaffold: history conditioning must still learn; its zero
initialization is not evidence that a new temporal predictor has been learned.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class ShallowStateUpdate(nn.Module):
    radius=4

    def __init__(self, blocks):
        super().__init__()
        if len(blocks)!=2:
            raise ValueError('Expected two mature128-channel residual blocks')
        for block in blocks:
            for conv in (block.conv1,block.conv2):
                if (conv.in_channels!=128 or conv.out_channels!=128 or conv.kernel_size!=(3,3)
                        or conv.stride!=(1,1) or conv.padding!=(1,1) or conv.dilation!=(1,1) or conv.groups!=1):
                    raise ValueError('Unsupported spatial update dependency')
            if block.downsample is not None:
                raise ValueError('Update must stay on one lattice')
        self.mix=nn.Conv2d(257,128,1,bias=False)
        self.blocks=copy.deepcopy(nn.Sequential(*blocks))
        self.mix.to(device=blocks[0].conv1.weight.device,dtype=blocks[0].conv1.weight.dtype)
        with torch.no_grad():
            self.mix.weight.zero_()
            for c in range(128):self.mix.weight[c,c,0,0]=1

    def forward(self, packed):
        return self.blocks(self.mix(packed))


class ShallowStateEncoder(nn.Module):
    def __init__(self, source, bounds, threshold=.5, period=3, execution='sparse'):
        super().__init__()
        if len(source)!=3 or threshold<0 or period<1 or execution not in ('sparse','masked_dense'):
            raise ValueError('Expected mature three-block stage and valid inference policy')
        self.stem=copy.deepcopy(source[0]);self.update=ShallowStateUpdate(list(source)[1:])
        self.bounds=bounds;self.threshold=threshold;self.period=period;self.execution=execution
        self.eval()

    def step(self,temporal,state,meta,reset=False,dense=False):
        if self.training or torch.is_grad_enabled():
            raise ValueError('This initial audit uses frozen inference only')
        current=self.stem(temporal);desc=early_descriptor(current,1)
        refresh=reset or dense or state is None
        if state is not None:
            elapsed=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
            if elapsed<=0 and not reset:raise ValueError('Non-causal spatial state')
            refresh=refresh or state['scene']!=meta['segment'] or elapsed>.25 or state['frames']>=self.period-1
        if refresh:
            old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            value=self.update(torch.cat((current,old,valid.to(current.dtype)),1));reference=desc
            stats=dict(actual_write_fraction=1.,executed_input_area_ratio=1.,fallback=True)
        else:
            aligned,valid=warp_bev(torch.cat((state['value'],state['reference']),1),state['pose'],meta['pose'],self.bounds)
            old,reference=aligned.split((128,16),dim=1)
            score=((desc-reference).square().mean(1,keepdim=True)/torch.maximum(
                desc.square().mean(1,keepdim=True),reference.square().mean(1,keepdim=True)).clamp_min(1e-6)).sqrt()
            # Cheap changes flag output tiles; refresh_tiles includes all four
            # convolution dependencies as halos rather than dilating writes too.
            request=(score>self.threshold)|~valid
            packed=torch.cat((current,old,valid.to(current.dtype)),1)
            value,written,stats=refresh_tiles(self.update,packed,old,request,1,4,tile=8,
                allow_fallback=True,fallback_fraction=.8,execution=self.execution)
            reference=torch.where(written,desc,reference)
        new=dict(value=value.detach(),reference=reference.detach(),pose=meta['pose'],scene=meta['segment'],
            timestamp_us=int(meta['timestamp_us']),frames=0 if refresh else state['frames']+1)
        stats.update(refresh=refresh,state_bytes=sum(new[k].numel()*new[k].element_size() for k in ('value','reference')))
        return value,new,stats

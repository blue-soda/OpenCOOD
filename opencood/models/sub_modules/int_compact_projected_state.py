"""Cache projected intermediate activations plus complete residual outputs.

The next spatial convolution consumes P(ReLU(BN(...))), so retaining that
projection avoids carrying the larger intermediate and reprojecting static
cells. Weights, nonlinearities and residual-block outputs remain unchanged.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_state import _selected_bn, _write_rows
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3_rows


class CompactProjectedSpatialStage(ProjectedSpatialStage):
    def __init__(self, source):
        nn.Module.__init__(self)
        if not isinstance(source,ProjectedSpatialStage) or source.training:
            raise ValueError('Frozen projected spatial stage required')
        self.stem = copy.deepcopy(source.stem)
        self.blocks = copy.deepcopy(source.blocks)
        self.rank = source.rank
        self.spatial_input_projected = True
        self.compact_projected_state = True
        self.allow_cache_grad = False
        self.retained_weight_energy = list(source.retained_weight_energy)
        self.cache_channels = []
        for block in self.blocks:
            if block.conv2.project.bias is not None:
                raise ValueError('Zero-bias projection required for zero-filled warp')
            self.cache_channels.extend((block.conv2.project.out_channels,block.conv2.mix.out_channels))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

    def dense_from_stem(self,current):
        maps = []
        for block in self.blocks:
            first = F.relu(block.bn1(block.conv1(current)))
            projected = block.conv2.project(first)
            current = F.relu(block.bn2(block.conv2.mix(projected))+current)
            maps.extend((projected,current))
        return current,maps

    def selected_from_stem(self,current,aligned_maps,mask,execution='selected'):
        if self.training or (torch.is_grad_enabled() and
                not (self.allow_cache_grad and execution == 'masked_dense')):
            raise ValueError('Frozen inference required')
        if execution not in ('selected','masked_dense') or len(aligned_maps)!=4:
            raise ValueError('Four compatible cached maps required')
        if current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or mask.dtype!=torch.bool:
            raise ValueError('Single-frame boolean mask required')
        for old,channels in zip(aligned_maps,self.cache_channels):
            if old.shape!=(1,channels,*current.shape[-2:]) or old.dtype!=current.dtype or old.device!=current.device:
                raise ValueError('Incompatible compact state')
        maps = []; macs = 0
        for index,block in enumerate(self.blocks):
            old_middle,old_out = aligned_maps[2*index:2*index+2]
            if execution=='masked_dense':
                first = F.relu(block.bn1(block.conv1(current)))
                projected = torch.where(mask,block.conv2.project(first),old_middle)
                current = torch.where(mask,F.relu(block.bn2(block.conv2.mix(projected))+current),old_out)
            else:
                rows,ids,count = block.conv1.selected(current,mask); macs += count
                if ids.numel():
                    activated = F.relu(_selected_bn(rows,block.bn1))
                    projected_rows = F.linear(activated,block.conv2.project.weight[:,:,0,0])
                    macs += ids.numel()*block.conv2.project.weight.numel()
                    projected = _write_rows(old_middle,ids,projected_rows)
                else:
                    projected = old_middle.clone()
                rows,ids,count = selected_conv3x3_rows(block.conv2.mix,projected,mask); macs += count
                if ids.numel():
                    skip = current.reshape(current.shape[1],-1)[:,ids].t()
                    current = _write_rows(old_out,ids,F.relu(_selected_bn(rows,block.bn2)+skip))
                else:
                    current = old_out.clone()
            maps.extend((projected,current))
        return current,maps,macs

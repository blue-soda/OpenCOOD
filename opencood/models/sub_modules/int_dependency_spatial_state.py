"""Exact-current local recomputation with nested receptive-field supports.

For a requested final mask M, four 3x3 layers execute dilate(M,3/2/1/0).
Updated final cells see only current stem evidence; cached values cannot enter
their dependency cone. Other cells retain their own aligned full activations.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_state import _selected_bn,_write_rows
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3_rows


def dependency_masks(core):
    if core.ndim!=4 or core.shape[:2]!=(1,1) or core.dtype!=torch.bool:
        raise ValueError('Single-frame boolean core mask required')
    masks=[core]
    for _ in range(3):masks.insert(0,F.max_pool2d(masks[0].float(),3,1,1)>0)
    return masks


class DependencyProjectedSpatialStage(CompactProjectedSpatialStage):
    def __init__(self,source):
        super().__init__(source)
        self.nested_dependency_masks=True
        self.last_layer_cells=None

    def selected_from_stem(self,current,aligned_maps,mask,execution='selected'):
        if self.training or torch.is_grad_enabled():raise ValueError('Frozen dependency inference required')
        if execution not in ('selected','masked_dense') or len(aligned_maps)!=4:
            raise ValueError('Four compatible maps required')
        if current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]):
            raise ValueError('Matching single-frame core required')
        for old,channels in zip(aligned_maps,self.cache_channels):
            if old.shape!=(1,channels,*current.shape[-2:]) or old.dtype!=current.dtype or old.device!=current.device:
                raise ValueError('Incompatible projected cache')
        masks=dependency_masks(mask);self.last_layer_cells=[int(m.sum()) for m in masks]
        maps=[];macs=0
        for index,block in enumerate(self.blocks):
            middle_mask,out_mask=masks[2*index:2*index+2]
            old_middle,old_out=aligned_maps[2*index:2*index+2]
            if execution=='masked_dense':
                first=F.relu(block.bn1(block.conv1(current)))
                projected=torch.where(middle_mask,block.conv2.project(first),old_middle)
                current=torch.where(out_mask,F.relu(block.bn2(block.conv2.mix(projected))+current),old_out)
            else:
                rows,ids,count=block.conv1.selected(current,middle_mask);macs+=count
                if ids.numel():
                    activated=F.relu(_selected_bn(rows,block.bn1))
                    values=F.linear(activated,block.conv2.project.weight[:,:,0,0])
                    macs+=ids.numel()*block.conv2.project.weight.numel()
                    projected=_write_rows(old_middle,ids,values)
                else:projected=old_middle.clone()
                rows,ids,count=selected_conv3x3_rows(block.conv2.mix,projected,out_mask);macs+=count
                if ids.numel():
                    skip=current.reshape(current.shape[1],-1)[:,ids].t()
                    current=_write_rows(old_out,ids,F.relu(_selected_bn(rows,block.bn2)+skip))
                else:current=old_out.clone()
            maps.extend((projected,current))
        return current,maps,macs

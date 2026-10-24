"""Dependency-complete single-H updates using only temporary active rows.

Four 3x3 outputs use supports3/2/1/0; their 1x1 inputs use4/3/2/1.
No full intermediate activation canvas or persistent intermediate is needed.
Index tables, patch gathering and final H copying remain real runtime costs.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_single_spatial_state import SingleSpatialStateStage
from opencood.models.sub_modules.int_dependency_spatial_state import dependency_masks
from opencood.models.sub_modules.int_basis_spatial_state import _selected_bn, _write_rows
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods


def row_lookup(ids, cells):
    table=ids.new_full((cells+1,),ids.numel())
    table[ids]=torch.arange(ids.numel(),device=ids.device)
    return table


def row_conv(conv, rows, ids, output_mask, chunk_size=4096):
    target,neighbors,shape=prepare_selected_neighborhoods(output_mask)
    cells=shape[-2]*shape[-1]
    lookup=row_lookup(ids,cells)
    indices=lookup[neighbors]
    # Missing in-grid dependencies are a bug, never silently zero-pad them.
    assert bool(((neighbors==cells)|(indices<ids.numel())).all())
    padded=torch.cat((rows,rows.new_zeros(1,rows.shape[1])),0)
    output=rows.new_empty(target.numel(),conv.out_channels)
    weight=conv.weight.permute(0,2,3,1).reshape(conv.out_channels,-1)
    for start in range(0,target.numel(),chunk_size):
        local=indices[start:start+chunk_size]
        patches=padded[local.reshape(-1)].reshape(local.shape[0],-1)
        output[start:start+local.shape[0]]=F.linear(patches,weight,conv.bias)
    return output,target,target.numel()*conv.weight.numel()


class RowSpatialStateStage(SingleSpatialStateStage):
    def __init__(self,source):
        super().__init__(source)
        self.row_spatial_execution=True
        self.last_projection_cells=None

    def selected_from_stem(self,current,aligned_maps,mask,execution='selected'):
        if execution=='masked_dense':
            return super().selected_from_stem(current,aligned_maps,mask,execution)
        if self.training or torch.is_grad_enabled() or execution!='selected':
            raise ValueError('Frozen selected row inference required')
        if current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or mask.dtype!=torch.bool:
            raise ValueError('Matching single-frame boolean mask required')
        if len(aligned_maps)!=1 or aligned_maps[0].shape!=current.shape:
            raise ValueError('One matching complete H required')
        old=aligned_maps[0]
        if old.dtype!=current.dtype or old.device!=current.device or mask.device!=current.device:
            raise ValueError('State, input and mask must share dtype/device')
        supports=dependency_masks(mask)
        input_support=F.max_pool2d(supports[0].float(),3,1,1)>0
        self.last_layer_cells=[int(m.sum()) for m in supports]
        self.last_projection_cells=[int(input_support.sum())]+self.last_layer_cells[:3]
        ids=input_support.reshape(-1).nonzero(as_tuple=False).flatten()
        rows=current[0].reshape(current.shape[1],-1)[:,ids].t()
        cells=mask.numel();macs=0
        for index,block in enumerate(self.blocks):
            projected=F.linear(rows,block.conv1.project.weight[:,:,0,0])
            macs+=ids.numel()*block.conv1.project.weight.numel()
            middle,middle_ids,count=row_conv(block.conv1.mix,projected,ids,supports[2*index]);macs+=count
            middle=F.relu(_selected_bn(middle,block.bn1))
            projected=F.linear(middle,block.conv2.project.weight[:,:,0,0])
            macs+=middle_ids.numel()*block.conv2.project.weight.numel()
            output,out_ids,count=row_conv(block.conv2.mix,projected,middle_ids,supports[2*index+1]);macs+=count
            skip_indices=row_lookup(ids,cells)[out_ids]
            assert bool((skip_indices<ids.numel()).all())
            rows=F.relu(_selected_bn(output,block.bn2)+rows[skip_indices])
            ids=out_ids
        output=_write_rows(old,ids,rows)
        return output,[output],macs

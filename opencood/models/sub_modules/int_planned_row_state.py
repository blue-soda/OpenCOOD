"""One per-frame dependency plan for single-H active-row execution.

Only indices are shared between layers. All current activations are ephemeral;
weights are packed once from the fixed checkpoint, without arithmetic changes.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_dependency_spatial_state import dependency_masks
from opencood.models.sub_modules.int_basis_spatial_state import _selected_bn, _write_rows
from opencood.models.sub_modules.int_row_spatial_state import RowSpatialStateStage, row_lookup, row_project


def dependency_plan(mask):
    supports=dependency_masks(mask)
    supports.insert(0,F.max_pool2d(supports[0].float(),3,1,1)>0)
    ids=[m.reshape(-1).nonzero(as_tuple=False).flatten() for m in supports]
    h,w=mask.shape[-2:];cells=h*w
    lookups=[row_lookup(locations,cells) for locations in ids[:-1]]
    offsets=torch.arange(-1,2,device=mask.device)
    neighbors=[];checks=[]
    for index,target in enumerate(ids[1:]):
        yy=torch.div(target[:,None,None],w,rounding_mode='floor')+offsets[None,:,None]
        xx=target[:,None,None]%w+offsets[None,None,:]
        valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
        locations=torch.where(valid,yy*w+xx,target.new_full((),cells)).reshape(-1,9)
        selected=lookups[index][locations]
        checks.append(((locations==cells)|(selected<ids[index].numel())).all())
        neighbors.append(selected)
    skips=[lookups[index][ids[index+2]] for index in (0,2)]
    checks.extend((value<ids[index].numel()).all() for index,value in zip((0,2),skips))
    assert bool(torch.stack(checks).all()), 'Missing current-frame dependency'
    return ids,neighbors,skips


def planned_conv(conv, weight, rows, indices, chunk_size=4096):
    padded=torch.cat((rows,rows.new_zeros(1,rows.shape[1])),0)
    output=rows.new_empty(indices.shape[0],conv.out_channels)
    for start in range(0,indices.shape[0],chunk_size):
        local=indices[start:start+chunk_size]
        patches=padded[local.reshape(-1)].reshape(local.shape[0],-1)
        output[start:start+local.shape[0]]=F.linear(patches,weight,conv.bias)
    return output


class PlannedRowSpatialStage(RowSpatialStateStage):
    def __init__(self,source):
        super().__init__(source)
        self.planned_row_execution=True
        for index,conv in enumerate(c for b in self.blocks for c in (b.conv1,b.conv2)):
            self.register_buffer('mix_{}_packed_weight'.format(index),
                conv.mix.weight.detach().permute(0,2,3,1).reshape(conv.mix.out_channels,-1).contiguous())

    def selected_from_stem(self,current,aligned_maps,mask,execution='selected'):
        if execution=='masked_dense':return super().selected_from_stem(current,aligned_maps,mask,execution)
        if self.training or torch.is_grad_enabled() or execution!='selected':
            raise ValueError('Frozen planned-row inference required')
        if current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or mask.dtype!=torch.bool:
            raise ValueError('Matching single-frame boolean mask required')
        if len(aligned_maps)!=1 or aligned_maps[0].shape!=current.shape:
            raise ValueError('One matching historical H required')
        old=aligned_maps[0]
        if old.dtype!=current.dtype or old.device!=current.device or mask.device!=current.device:
            raise ValueError('State, input and mask must share dtype/device')
        ids,neighbors,skips=dependency_plan(mask)
        self.last_layer_cells=[v.numel() for v in ids[1:]]
        self.last_projection_cells=[v.numel() for v in ids[:-1]]
        rows=current[0].reshape(current.shape[1],-1)[:,ids[0]].t()
        macs=0
        for index,block in enumerate(self.blocks):
            projected=row_project(rows,block.conv1.project)
            first=planned_conv(block.conv1.mix,getattr(self,'mix_{}_packed_weight'.format(2*index)),projected,neighbors[2*index])
            middle=F.relu(_selected_bn(first,block.bn1))
            projected=row_project(middle,block.conv2.project)
            output=planned_conv(block.conv2.mix,getattr(self,'mix_{}_packed_weight'.format(2*index+1)),projected,neighbors[2*index+1])
            rows=F.relu(_selected_bn(output,block.bn2)+rows[skips[index]])
            for j,conv in enumerate((block.conv1,block.conv2)):
                k=2*index+j
                macs+=ids[k].numel()*conv.project.weight.numel()+ids[k+1].numel()*conv.mix.weight.numel()
        output=_write_rows(old,ids[-1],rows)
        return output,[output],macs

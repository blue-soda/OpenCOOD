"""Reuse complete detector features; recompute two-convolution local readout.

Selection compares its actual current input (shallow + corrected deep), not
the upstream H mask. Functional kernels skip unrequested output arithmetic.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods, prepared_conv3x3_rows
from opencood.models.sub_modules.int_row_spatial_state import row_lookup
from opencood.models.sub_modules.int_planned_row_state import planned_conv
from opencood.models.sub_modules.int_basis_spatial_state import _write_rows, _selected_bn


class LocalReadoutCache:
    def __init__(self, readout, bounds, threshold, execution='selected', period=3,row_chunk_size=4096):
        if readout.training or execution not in ('selected', 'masked_dense') or period not in (1,3,5):
            raise ValueError('Frozen readout, selected/masked_dense and period1/3/5 required')
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('Finite nonnegative threshold required')
        if not isinstance(row_chunk_size,int) or row_chunk_size<1:raise ValueError('Positive row chunk size required')
        self.row_chunk_size=row_chunk_size
        # CoDynTrust-derived checkpoint: BN/ReLU -> (1x1,3x3) -> BN/ReLU.
        tail = readout.tail
        if (len(tail)!=5 or not isinstance(tail[0],nn.BatchNorm2d) or
                not isinstance(tail[1],nn.ReLU) or not isinstance(tail[2],nn.Sequential) or len(tail[2])!=2 or
                not isinstance(tail[3],nn.BatchNorm2d) or not isinstance(tail[4],nn.ReLU)):
            raise ValueError('Expected compact DoubleConv readout')
        self.first, self.project, self.second = readout.spatial, tail[2][0], tail[2][1]
        self.bn1,self.bn2=tail[0],tail[3]
        for conv in (self.first,self.second):
            if (not isinstance(conv,nn.Conv2d) or conv.kernel_size!=(3,3) or
                    conv.padding!=(1,1) or conv.stride!=(1,1) or conv.dilation!=(1,1) or
                    conv.groups!=1 or conv.padding_mode!='zeros'):
                raise ValueError('Expected ordinary zero-padded 3x3')
        if (not isinstance(self.project,nn.Conv2d) or self.project.kernel_size!=(1,1) or
                self.project.stride!=(1,1) or self.project.padding!=(0,0) or self.project.groups!=1):
            raise ValueError('Expected unit-stride 1x1 projection')
        self.readout,self.bounds,self.threshold,self.execution,self.period=readout,bounds,threshold,execution,period
        self.max_full_age_us=450000 if period==5 else 250000
        self.weights=[c.weight.detach().permute(0,2,3,1).reshape(c.out_channels,-1).contiguous()
                      for c in (self.first,self.second)]
        self.last_mask=None

    @property
    def packed_weight_bytes(self):
        return sum(w.numel()*w.element_size() for w in self.weights)

    def dense(self, current):
        return self.readout.tail(self.first(current))

    def selected(self, current, old, mask):
        if torch.is_grad_enabled() or self.readout.training:
            raise ValueError('Frozen inference required')
        if (current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or
                mask.dtype!=torch.bool or old.shape!=(1,self.second.out_channels,*current.shape[-2:])):
            raise ValueError('Matching single-frame input, output cache and boolean mask required')
        support=F.max_pool2d(mask.float(),3,1,1)>0
        plan=prepare_selected_neighborhoods(support)
        middle,ids,macs=prepared_conv3x3_rows(self.first,self.weights[0],current,plan,self.row_chunk_size)
        middle=F.relu(_selected_bn(middle,self.bn1))
        projected=F.linear(middle,self.project.weight[:,:,0,0],self.project.bias)
        target,neighbors,_=prepare_selected_neighborhoods(mask)
        indices=row_lookup(ids,mask.numel())[neighbors]
        assert bool(((neighbors==mask.numel())|(indices<ids.numel())).all())
        output=F.relu(_selected_bn(planned_conv(self.second,self.weights[1],projected,indices,self.row_chunk_size),self.bn2))
        macs+=ids.numel()*self.project.weight.numel()+target.numel()*self.second.weight.numel()
        return _write_rows(old,target,output),macs,ids.numel()

    def select_mask(self,current,reference,valid,confidence):
        changed=local_change_mask(current,reference,valid,self.threshold)
        mask=F.max_pool2d(changed.float(),5,1,2)>0
        priority=F.max_pool2d((confidence>=.1).float(),9,1,4)>0
        return mask|priority|~valid

    def step(self, current, state, meta, reset=False):
        if torch.is_grad_enabled() or self.readout.training:
            raise ValueError('Frozen inference required')
        stamp=int(meta['timestamp_us'])
        reason='reset' if reset or state is None or state['scene']!=meta['segment'] else None
        if reason is None:
            if stamp<=state['stamp']: raise ValueError('Noncausal readout stream')
            if stamp-state['full_stamp']>=self.max_full_age_us: reason='age'
            elif state['frames']+1>=self.period: reason='period'
        extra=0; uncovered=0.
        if reason:
            output=self.dense(current); reference=current
            mask=torch.ones_like(current[:,:1],dtype=torch.bool)
            support_count=mask.numel(); frames=0; full_stamp=stamp
        else:
            packed=torch.cat((state['value'],state['reference'],state['confidence']),1)
            aligned,valid=warp_bev(packed,state['pose'],meta['pose'],self.bounds)
            co=self.second.out_channels; ci=current.shape[1]
            old,reference=aligned[:,:co],aligned[:,co:co+ci]
            mask=self.select_mask(current,reference,valid,aligned[:,-1:])
            assert bool((mask|valid).all())
            if self.execution=='selected':
                output,extra,support_count=self.selected(current,old,mask)
            else:
                output=torch.where(mask,self.dense(current),old)
                support_count=mask.numel()
            reference=torch.where(mask,current,reference)
            frames=state['frames']+1;full_stamp=state['full_stamp']
            uncovered=float((~valid).float().mean())
        self.last_mask=mask
        new=dict(value=output.detach(),reference=reference.detach(),pose=meta['pose'],
                 stamp=stamp,full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        count=int(mask.sum());cells=mask.numel()
        weight1=self.first.weight.numel()+self.project.weight.numel();weight2=self.second.weight.numel()
        selected=not reason and self.execution=='selected'
        info=dict(readout_refresh=reason is not None,readout_refresh_reason=reason,
            readout_write_fraction=count/float(cells),readout_uncovered_fraction=uncovered,
            readout_executed_fraction=extra/float(cells*(weight1+weight2)) if selected else 1.,
            readout_extra_linear_macs=extra,readout_full_age_s=(stamp-full_stamp)/1e6,
            readout_state_bytes=sum(v.numel()*v.element_size() for v in (new['value'],new['reference'])))
        return output,new,info

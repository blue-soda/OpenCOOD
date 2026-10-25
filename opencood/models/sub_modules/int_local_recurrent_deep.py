"""Complete D reuse outside the actual H update support; short repair inside."""
import copy
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepState,RecurrentDeepDetector
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods
from opencood.models.sub_modules.int_row_spatial_state import row_lookup
from opencood.models.sub_modules.int_basis_spatial_state import _write_rows


def local_repair(repair,h,old,valid,mask,execution='selected'):
    if torch.is_grad_enabled() or repair.training:raise ValueError('Frozen inference required')
    if execution not in ('selected','masked_dense'):raise ValueError('Unknown local repair execution')
    if h.shape[0]!=1 or h.shape[1]!=128 or old.shape[0]!=1 or old.shape[1]!=repair.channels or h.shape[-2:]!=old.shape[-2:]:
        raise ValueError('Single matching H and D required')
    if mask.shape!=old[:,:1].shape or mask.dtype!=torch.bool or valid.shape!=mask.shape or valid.dtype!=torch.bool:
        raise ValueError('Matching boolean selection and validity required')
    first,dw,last=repair.net[0],repair.net[2],repair.net[4]
    assert first.kernel_size==last.kernel_size==(1,1) and dw.kernel_size==(3,3) and dw.groups==dw.in_channels==dw.out_channels
    cells=mask.numel();count=int(mask.sum());full_macs=cells*(first.weight.numel()+dw.weight.numel()+last.weight.numel())
    if execution=='masked_dense' or count==cells:
        value=repair(torch.cat((h,old,valid.to(old.dtype)),1))
        if count!=cells:value=torch.where(mask,value,old)
        return value,dict(deep_extra_linear_macs=0,deep_macs_saved=0,deep_write_fraction=count/float(cells),deep_short_macs_ratio=1.,deep_input_fraction=1.)
    if count==0:
        return old,dict(deep_extra_linear_macs=0,deep_macs_saved=full_macs,deep_write_fraction=0.,deep_short_macs_ratio=0.,deep_input_fraction=0.)
    support=F.max_pool2d(mask.float(),3,1,1)>0
    ids=support.reshape(-1).nonzero(as_tuple=False).flatten()
    rows=torch.cat([x[0].reshape(x.shape[1],-1)[:,ids].t() for x in (h,old,valid.to(old.dtype))],1)
    projected=F.relu(F.linear(rows,first.weight[:,:,0,0],first.bias))
    target,neighbors,shape=prepare_selected_neighborhoods(mask)
    indices=row_lookup(ids,cells)[neighbors]
    assert bool(((neighbors==cells)|(indices<ids.numel())).all()),'Missing current input halo'
    padded=torch.cat((projected,projected.new_zeros(1,projected.shape[1])),0)
    patches=padded[indices.reshape(-1)].reshape(target.numel(),9,dw.in_channels)
    middle=(patches*dw.weight[:,0].reshape(dw.out_channels,9).t().unsqueeze(0)).sum(1)
    middle=F.relu(middle+dw.bias)
    correction=F.linear(middle,last.weight[:,:,0,0],last.bias)
    value=_write_rows(old,target,old[0].reshape(old.shape[1],-1)[:,target].t()+correction)
    macs=ids.numel()*first.weight.numel()+target.numel()*(dw.weight.numel()+last.weight.numel())
    return value,dict(deep_extra_linear_macs=macs,deep_macs_saved=full_macs-macs,deep_write_fraction=count/float(cells),
        deep_short_macs_ratio=macs/float(full_macs),deep_input_fraction=ids.numel()/float(cells))


class LocalRecurrentDeepState(RecurrentDeepState):
    def __init__(self,repair,bounds,period=4,execution='selected'):
        super().__init__(repair,bounds,period);self.execution=execution

    def update(self,h,old,valid,update_mask=None):
        if update_mask is None:raise ValueError('Actual current H update mask required')
        # A changed H cell can affect repair outputs at distance1. The repair
        # then needs its own input halo1. Uncovered D is always recomputed.
        core=(F.max_pool2d(update_mask.float(),3,1,1)>0)|~valid
        return local_repair(self.repair,h,old,valid,core,self.execution)


class LocalRecurrentDeepDetector(RecurrentDeepDetector):
    def __init__(self,baseline,repair,period=4,execution='selected'):
        super().__init__(baseline,repair,period)
        self.spatial=copy.copy(self.spatial);self.spatial.capture_update_mask=True
        self.deep=LocalRecurrentDeepState(repair,self.bounds,period,execution)

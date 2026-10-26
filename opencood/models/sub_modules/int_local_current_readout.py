"""Complete Y reuse with genuinely local two-spatial-layer current prediction."""
import math
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods,prepared_conv3x3_rows
from opencood.models.sub_modules.int_row_spatial_state import row_lookup
from opencood.models.sub_modules.int_planned_row_state import planned_conv
from opencood.models.sub_modules.int_basis_spatial_state import _write_rows
from opencood.models.sub_modules.int_cached_neighborhoods import CachedNeighborhoods


class LocalCurrentReadoutCache:
    def __init__(self,readout,repair,bounds,threshold,backend='selected',period=3,row_chunk_size=32768,cache_geometry=False):
        if readout.training or repair.training or repair.gated or backend not in ('selected','masked_dense'):
            raise ValueError('Frozen current-only readout and selected/masked_dense backend required')
        if period not in (1,3,5) or not math.isfinite(threshold) or threshold<0:raise ValueError('Invalid refresh/threshold')
        if not isinstance(row_chunk_size,int) or row_chunk_size<1:raise ValueError('Positive row chunk size required')
        self.row_chunk_size=row_chunk_size
        self.geometry=CachedNeighborhoods() if cache_geometry else None
        self.readout,self.repair,self.bounds,self.threshold=readout,repair,bounds,threshold
        self.backend,self.period=backend,period
        self.execution='current_local';self.max_full_age_us=450000 if period==5 else 250000
        self.first,self.second,self.output=repair.net[0],repair.net[2],repair.net[3]
        self.weights=[m.weight.detach().permute(0,2,3,1).reshape(m.out_channels,-1).contiguous() for m in (self.first,self.second)]
        self.last_mask=None

    @property
    def packed_weight_bytes(self):
        return sum(w.numel()*w.element_size() for w in self.weights)

    def dense(self,current):
        return self.repair.net(current)

    @property
    def geometry_bytes(self):
        return self.geometry.geometry_bytes if self.geometry is not None else 0

    def neighborhood_plan(self,mask):
        return self.geometry(mask) if self.geometry is not None else prepare_selected_neighborhoods(mask)

    def select_mask(self,current,reference,valid,confidence):
        return LocalReadoutCache.select_mask(self,current,reference,valid,confidence)

    def selected(self,current,old,mask):
        if torch.is_grad_enabled() or self.repair.training:raise ValueError('Frozen inference only')
        if (current.shape[0]!=1 or mask.shape!=(1,1,*current.shape[-2:]) or mask.dtype!=torch.bool or
                old.shape!=(1,self.repair.channels,*current.shape[-2:])):
            raise ValueError('Matching single-frame input, output cache and boolean mask required')
        cells=mask.numel();cost=sum(m.weight.numel() for m in (self.first,self.second,self.output))
        if bool(mask.all()):return self.dense(current),cells*cost,0,cells
        if not bool(mask.any()):return old,0,0,0
        output,target,macs,support=self.partial_rows(current,mask)
        return _write_rows(old,target,output),macs,macs,support

    def partial_rows(self,current,mask):
        """Compute requested rows; caller owns the final complete-state write."""
        cells=mask.numel()
        support=F.max_pool2d(mask.float(),3,1,1)>0
        middle,ids,macs=prepared_conv3x3_rows(self.first,self.weights[0],current,self.neighborhood_plan(support),self.row_chunk_size)
        middle=F.relu(middle)
        target,neighbors,_=self.neighborhood_plan(mask)
        indices=row_lookup(ids,cells)[neighbors]
        assert bool(((neighbors==cells)|(indices<ids.numel())).all())
        output=planned_conv(self.second,self.weights[1],middle,indices,self.row_chunk_size)
        output=F.relu(F.linear(output,self.output.weight[:,:,0,0],self.output.bias))
        macs+=target.numel()*(self.second.weight.numel()+self.output.weight.numel())
        return output,target,macs,ids.numel()

    def step(self,current,state,meta,reset=False):
        if torch.is_grad_enabled() or self.repair.training:raise ValueError('Frozen inference only')
        stamp=int(meta['timestamp_us']);reason='reset' if reset or state is None or state['scene']!=meta['segment'] else None
        if reason is None:
            if stamp<=state['stamp']:raise ValueError('Noncausal readout state')
            if stamp-state['full_stamp']>=self.max_full_age_us:reason='age'
            elif state['frames']+1>=self.period:reason='period'
        cells=current.shape[-2]*current.shape[-1]
        mature=cells*sum(c.weight.numel() for c in (self.readout.spatial,self.readout.tail[2][0],self.readout.tail[2][1]))
        extra=0;uncovered=0.
        if reason:
            output=self.readout.tail(self.readout.spatial(current));reference=current
            mask=torch.ones_like(current[:,:1],dtype=torch.bool);support=cells;executed=mature
            frames=0;full_stamp=stamp
        else:
            aligned,valid=warp_bev(torch.cat((state['value'],state['reference'],state['confidence']),1),state['pose'],meta['pose'],self.bounds)
            old=aligned[:,:self.repair.channels];reference=aligned[:,self.repair.channels:-1]
            mask=self.select_mask(current,reference,valid,aligned[:,-1:])
            assert bool((mask|valid).all())
            if self.backend=='selected':output,executed,extra,support=self.selected(current,old,mask)
            else:
                output=torch.where(mask,self.dense(current),old);support=cells
                executed=cells*sum(c.weight.numel() for c in (self.first,self.second,self.output))
            reference=torch.where(mask,current,reference)
            frames=state['frames']+1;full_stamp=state['full_stamp'];uncovered=float((~valid).float().mean())
        self.last_mask=mask
        new=dict(value=output.detach(),reference=reference.detach(),pose=meta['pose'],stamp=stamp,
            full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        info=dict(readout_refresh=reason is not None,readout_refresh_reason=reason,readout_depth=frames,
            readout_write_fraction=float(mask.float().mean()),readout_executed_fraction=executed/float(mature),
            readout_macs_saved=mature-executed,readout_extra_linear_macs=extra,readout_executed_macs=executed,
            readout_uncovered_fraction=uncovered,readout_first_support_fraction=support/float(cells),
            readout_state_bytes=sum(v.numel()*v.element_size() for v in (new['value'],new['reference'])))
        return output,new,info

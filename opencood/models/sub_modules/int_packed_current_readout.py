"""Current readout with one owned packed complete-state allocation per step."""
import torch
from opencood.models.sub_modules.int_local_current_readout import LocalCurrentReadoutCache
from opencood.models.sub_modules.int_cached_metric_warp import CachedMetricWarp


class PackedCurrentReadoutCache(LocalCurrentReadoutCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.warp=CachedMetricWarp(self.bounds)

    @property
    def geometry_bytes(self):
        return super().geometry_bytes+self.warp.geometry_bytes

    def set_confidence(self,state,confidence):
        if confidence.shape!=state['confidence'].shape:raise ValueError('Matching own confidence required')
        state['confidence'].copy_(confidence.detach());state['confidence_ready']=True

    def selected_into(self,current,owned,mask):
        """Write only into this step's fresh warp allocation, never prior state."""
        cells=mask.numel()
        if bool(mask.all()):
            owned.copy_(self.dense(current))
            return cells*sum(c.weight.numel() for c in (self.first,self.second,self.output)),0,cells
        if not bool(mask.any()):return 0,0,0
        rows,ids,macs,support=self.partial_rows(current,mask)
        owned[0].reshape(self.repair.channels,-1)[:,ids]=rows.t()
        return macs,macs,support

    def step(self,current,state,meta,reset=False):
        if torch.is_grad_enabled() or self.repair.training:raise ValueError('Frozen inference only')
        stamp=int(meta['timestamp_us']);co=self.repair.channels;ci=current.shape[1]
        reason='reset' if reset or state is None or state['scene']!=meta['segment'] else None
        if reason is None:
            if stamp<=state['stamp']:raise ValueError('Noncausal readout state')
            if stamp-state['full_stamp']>=self.max_full_age_us:reason='age'
            elif state['frames']+1>=self.period:reason='period'
        cells=current.shape[-2]*current.shape[-1]
        mature=cells*sum(c.weight.numel() for c in (self.readout.spatial,self.readout.tail[2][0],self.readout.tail[2][1]))
        extra=0;uncovered=0.
        if reason:
            value=self.readout.tail(self.readout.spatial(current))
            packed=torch.cat((value,current,torch.zeros_like(current[:,:1])),1)
            mask=torch.ones_like(current[:,:1],dtype=torch.bool);support=cells;executed=mature
            frames=0;full_stamp=stamp
        else:
            if not state.get('confidence_ready',False):raise ValueError('Attach own confidence before advancing')
            packed,valid=self.warp(state['packed'],state['pose'],meta['pose'],self.bounds)
            value,reference=packed[:,:co],packed[:,co:co+ci]
            mask=self.select_mask(current,reference,valid,packed[:,-1:]);assert bool((mask|valid).all())
            if self.backend=='selected':executed,extra,support=self.selected_into(current,value,mask)
            else:
                value.copy_(torch.where(mask,self.dense(current),value));support=cells
                executed=cells*sum(c.weight.numel() for c in (self.first,self.second,self.output))
            ids=mask.flatten().nonzero(as_tuple=False).flatten()
            reference[0].reshape(ci,-1)[:,ids]=current[0].reshape(ci,-1)[:,ids]
            frames=state['frames']+1;full_stamp=state['full_stamp'];uncovered=float((~valid).float().mean())
        self.last_mask=mask
        new=dict(packed=packed.detach(),value=packed[:,:co].detach(),reference=packed[:,co:co+ci].detach(),
            confidence=packed[:,-1:].detach(),confidence_ready=False,pose=meta['pose'],stamp=stamp,
            full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        info=dict(readout_refresh=reason is not None,readout_refresh_reason=reason,readout_depth=frames,
            readout_write_fraction=float(mask.float().mean()),readout_executed_fraction=executed/float(mature),
            readout_macs_saved=mature-executed,readout_extra_linear_macs=extra,readout_executed_macs=executed,
            readout_uncovered_fraction=uncovered,readout_first_support_fraction=support/float(cells),
            readout_state_bytes=(co+ci)*cells*packed.element_size())
        return new['value'],new,info


def pack_readout_state(state):
    """Same-input profiler adapter, outside timing; deployment never calls it."""
    if state is None or 'packed' in state:return state
    co=state['value'].shape[1];ci=state['reference'].shape[1]
    packed=torch.cat((state['value'],state['reference'],state['confidence']),1)
    new=dict(state);new.update(packed=packed,value=packed[:,:co],reference=packed[:,co:co+ci],
        confidence=packed[:,-1:],confidence_ready=True)
    return new

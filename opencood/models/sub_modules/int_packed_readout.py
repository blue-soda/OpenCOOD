"""One owned packed state, aligned once and updated only at selected cells.

Each step owns a fresh warp allocation. Previous state is never modified.
Output/reference/confidence are disjoint views, not duplicated activations.
"""
import torch
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.models.sub_modules.int_cached_metric_warp import CachedMetricWarp


class PackedReadoutCache(LocalReadoutCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.warp=CachedMetricWarp(self.bounds)

    @property
    def geometry_bytes(self):return self.warp.geometry_bytes

    def set_confidence(self,state,confidence):
        if confidence.shape!=state['confidence'].shape:raise ValueError('Matching confidence required')
        state['confidence'].copy_(confidence.detach());state['confidence_ready']=True

    def step(self,current,state,meta,reset=False):
        if torch.is_grad_enabled() or self.readout.training:raise ValueError('Frozen inference required')
        if current.shape[0]!=1 or current.shape[1]!=self.first.in_channels:raise ValueError('Single matching input required')
        stamp=int(meta['timestamp_us']);co=self.second.out_channels;ci=current.shape[1]
        reason='reset' if reset or state is None or state['scene']!=meta['segment'] else None
        if reason is None:
            if stamp<=state['stamp']:raise ValueError('Noncausal readout stream')
            if stamp-state['full_stamp']>=self.max_full_age_us:reason='age'
            elif state['frames']+1>=self.period:reason='period'
        extra=0;uncovered=0.
        if reason:
            output=self.dense(current)
            packed=torch.cat((output,current,torch.zeros_like(current[:,:1])),1)
            mask=torch.ones_like(current[:,:1],dtype=torch.bool);frames=0;full_stamp=stamp
        else:
            if not state.get('confidence_ready',False):raise ValueError('Attach own prediction confidence before next step')
            packed,valid=self.warp(state['packed'],state['pose'],meta['pose'],self.bounds)
            output,reference=packed[:,:co],packed[:,co:co+ci]
            mask=self.select_mask(current,reference,valid,packed[:,-1:]);assert bool((mask|valid).all())
            if self.execution=='selected':rows,target,extra,_=self.selected_rows(current,mask)
            else:
                target=mask.flatten().nonzero(as_tuple=False).flatten()
                rows=self.dense(current)[0].reshape(co,-1)[:,target].t()
            # These views belong to THIS step's fresh alignment allocation.
            output[0].reshape(co,-1)[:,target]=rows.t()
            reference[0].reshape(ci,-1)[:,target]=current[0].reshape(ci,-1)[:,target]
            frames=state['frames']+1;full_stamp=state['full_stamp'];uncovered=float((~valid).float().mean())
        self.last_mask=mask
        new=dict(packed=packed.detach(),value=packed[:,:co].detach(),reference=packed[:,co:co+ci].detach(),
            confidence=packed[:,-1:].detach(),confidence_ready=False,pose=meta['pose'],stamp=stamp,
            full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        cells=mask.numel();selected=reason is None and self.execution=='selected'
        full=cells*sum(c.weight.numel() for c in (self.first,self.project,self.second))
        info=dict(readout_refresh=reason is not None,readout_refresh_reason=reason,
            readout_write_fraction=int(mask.sum())/float(cells),readout_uncovered_fraction=uncovered,
            readout_executed_fraction=extra/float(full) if selected else 1.,readout_extra_linear_macs=extra,
            readout_full_age_s=(stamp-full_stamp)/1e6,
            readout_state_bytes=(co+ci)*cells*packed.element_size())
        return new['value'],new,info

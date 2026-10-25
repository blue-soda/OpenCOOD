"""Predict complete readout state with a cheap recursive correction.

Full spatial readout runs only on declared refresh frames in deployment.
During training, extra dense targets are discarded and never enter history.
"""
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev


class ShortReadoutState:
    def __init__(self,readout,repair,bounds,period=3):
        if period not in (1,3):raise ValueError('Period1/3 supported')
        self.readout,self.repair,self.bounds,self.period=readout,repair,bounds,period
        self.execution='short';self.training_targets=False;self.packed_weight_bytes=0
        self.max_full_age_us=250000

    def step(self,current,state,meta,reset=False):
        if torch.is_grad_enabled() and not self.training_targets:raise ValueError('Gradients require training targets')
        stamp=int(meta['timestamp_us']);reason='reset' if reset or state is None or state['scene']!=meta['segment'] else None
        if reason is None:
            if stamp<=state['stamp']:raise ValueError('Noncausal readout state')
            if stamp-state['full_stamp']>=self.max_full_age_us:reason='age'
            elif state['frames']+1>=self.period:reason='period'
        with torch.no_grad():
            target=self.readout.tail(self.readout.spatial(current)) if reason or self.training_targets else None
        if reason:value=target;frames=0;full_stamp=stamp;uncovered=0.
        else:
            old,valid=warp_bev(state['value'],state['pose'],meta['pose'],self.bounds)
            value=self.repair(torch.cat((current,old,valid.to(old.dtype)),1))
            frames=state['frames']+1;full_stamp=state['full_stamp'];uncovered=float((~valid).float().mean())
        new=dict(value=value if self.training_targets else value.detach(),pose=meta['pose'],stamp=stamp,
            full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        cells=current.shape[-2]*current.shape[-1]
        full=cells*sum(c.weight.numel() for c in (self.readout.spatial,self.readout.tail[2][0],self.readout.tail[2][1]))
        short=cells*sum(c.weight.numel() for c in (self.repair.net[0],self.repair.net[2],self.repair.net[4]))
        info=dict(readout_refresh=reason is not None,readout_refresh_reason=reason,readout_depth=frames,
            readout_write_fraction=1.,readout_executed_fraction=1. if reason else short/float(full),
            readout_macs_saved=0 if reason else full-short,readout_extra_linear_macs=0,
            readout_uncovered_fraction=uncovered,readout_state_bytes=value.numel()*value.element_size())
        if self.training_targets:info.update(readout_target=target,readout_student=value)
        return value,new,info

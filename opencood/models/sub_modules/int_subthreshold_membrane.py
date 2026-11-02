"""Single coarse task state with retained or discarded subthreshold information.

The learned ordinary gate and protected refresh candidate are unchanged. This is
a dense hybrid membrane/quantized-communication model, not a sparse SNN kernel.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector,reset_reason
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


def quantize_ste(value,quantum):
    if not quantum>0:raise ValueError('Positive quantum required')
    rounded=torch.round(value.detach()/quantum)*quantum
    return rounded+(value-value.detach())


class SubthresholdMembraneDetector(ProtectedMultiscaleRefreshDetector):
    def __init__(self,source,mode='carry',quantum=.25):
        nn.Module.__init__(self)
        if type(source) is not ProtectedMultiscaleRefreshDetector or mode not in ('continuous','carry','drop','reset') or quantum<=0:raise ValueError('Protected source and valid mode required')
        self.core=copy.deepcopy(source.core);self.pillar_vfe,self.scatter=source.pillar_vfe,source.scatter
        for key in ('bounds','memory_policy','spatial_only','period','max_age','light_observations'):setattr(self,key,copy.deepcopy(getattr(source,key)))
        self.membrane_mode,self.quantum=mode,float(quantum);self.variant='subthreshold_'+mode;self.eval()

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if spatial.shape[0]!=1 or requests is not None:raise ValueError('Single stream, no extra spatial mask')
        reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us']);refresh=reason
        if refresh is None:
            if state['since_refresh']+1>=self.period:refresh='period'
            elif (stamp-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6:refresh='age'
        x=spatial;values=[];new={};stats=None
        for key,encoder,light,update,stride in (
            ('value',self.core.observation,None,self.core.fine_update,2),
            ('coarse',self.core.coarse_encoder,self.core.coarse_light,self.core.coarse_update,4),
            ('deep',self.core.deep_encoder,self.core.deep_light,self.core.deep_update,8)):
            current=(encoder if refresh or light is None or not self.light_observations else light)(x)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:
                bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride)
                old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            if self.membrane_mode=='reset' and key=='coarse':
                old=torch.zeros_like(old);valid=torch.zeros_like(valid)
            if refresh and key in self.core.refresh_updates:update=self.core.refresh_updates[key]
            update.delta_t=dt;v=update(current,old,valid);x=v;stored=v
            if key=='coarse' and self.membrane_mode in ('carry','drop'):
                sent=quantize_ste(v,self.quantum);x=sent
                if self.membrane_mode=='drop':stored=sent
                residual=v.detach()-sent.detach()
                stats=dict(quantum=self.quantum,max_abs_residual=float(residual.abs().max()),residual_square_sum=float(residual.double().square().sum()),elements=residual.numel(),retains_subthreshold=self.membrane_mode=='carry',extra_persistent_tensors=0)
                assert stats['max_abs_residual']<=self.quantum*.5001
            new[key]=stored.detach() if detach_state else stored;values.append(x)
        features=self.core.readout(torch.cat([m(v) for m,v in zip(self.core.deblocks,values)],1))
        pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        size=sum(v.numel()*v.element_size() for v in new.values())
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,since_refresh=0 if refresh else state['since_refresh']+1,refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        return pred,new,dict(reset=reason,full_refresh=refresh,cache_bytes=size,membrane=stats,middle_deep_encoder_executed=bool(refresh) or not self.light_observations,state_updates_executed=True)

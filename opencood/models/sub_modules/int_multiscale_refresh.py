"""Trainable recurrent state and middle/deep refresh with light observations."""
import copy
import math
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_mechanism_probes import SingleGateStateUpdate
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class LeakyStateUpdate(nn.Module):
    """Continuous retention control with the same copied candidate, fixed tau."""
    def __init__(self,candidate,tau=.2):
        super().__init__();self.candidate=copy.deepcopy(candidate);self.tau=tau;self.delta_t=.1

    def forward(self,current,old,valid):
        if not 0<=self.delta_t<=.25:raise ValueError('Reset required across a gap')
        old=torch.where(valid,old,torch.zeros_like(old));candidate=self.candidate(current,old,valid)
        z=1-math.exp(-self.delta_t/self.tau)
        return torch.where(valid,old+z*(candidate-old),candidate)


def replace_update(candidate,channels,mode):
    if mode=='original':return copy.deepcopy(candidate)
    if mode=='gate':return SingleGateStateUpdate(candidate,channels)
    if mode=='leaky':return LeakyStateUpdate(candidate)
    raise ValueError(mode)


def reset_reason(state,meta):
    if state is None:return 'initial',0.
    if meta.get('reset_before',False):return 'explicit',0.
    if state['scene']!=meta['segment']:return 'scene',0.
    gap=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
    if gap<=0:raise ValueError('Noncausal state')
    return ('gap',0.) if gap>.25 else (None,gap)


class TemporalPyramidDetector(PyramidStateDetector):
    """Single coarse update intervention, retaining the complete spatial path."""
    def __init__(self,base,mode='gate'):
        nn.Module.__init__(self)
        if type(base) is not PyramidStateDetector or base.memory_policy!='recurrent':raise ValueError('Recurrent base required')
        self.core=copy.deepcopy(base.core)
        self.pillar_vfe,self.scatter=base.pillar_vfe,base.scatter
        self.bounds=copy.deepcopy(base.bounds);self.memory_policy='recurrent';self.variant='coarse_'+mode
        self.spatial_only=False
        self.core.coarse_update=replace_update(base.core.coarse_update,256,mode)
        self.update_mode=mode;self.train(False)

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if requests is not None:raise ValueError('No spatial masks in the temporal control')
        _,dt=reset_reason(state,meta)
        self.core.coarse_update.delta_t=dt
        return super().step_spatial(spatial,state,meta,detach_state=detach_state)


class MultiscaleRefreshDetector(TemporalPyramidDetector):
    """Full middle/deep encoding on refresh, learned light observations otherwise.

    Recurrent states really update each frame, so they align from the last frame,
    not an immutable anchor. Invalid history initializes from current observation.
    Ordinary paths require cache-aware training; random projections are not a
    pretrained accuracy-preserving cache replacement.
    """
    def __init__(self,base,mode='gate',period=4,max_age=.45):
        super().__init__(base,mode)
        if type(period) is not int or not 1<=period<=6 or not 0<max_age<=.65:raise ValueError('Invalid refresh schedule')
        self.period,self.max_age=period,max_age
        self.core.deep_update=replace_update(base.core.deep_update,512,mode)
        self.core.coarse_light=nn.Sequential(nn.Conv2d(128,128,3,stride=2,padding=1,groups=128),nn.ReLU(),nn.Conv2d(128,256,1))
        self.core.deep_light=nn.Sequential(nn.Conv2d(256,256,3,stride=2,padding=1,groups=256),nn.ReLU(),nn.Conv2d(256,512,1))
        self.variant='middle_deep_light_'+mode;self.train(False)

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if spatial.shape[0]!=1 or requests is not None:raise ValueError('Single stream; no local masks yet')
        reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us'])
        refresh=reason
        if refresh is None:
            if state['since_refresh']+1>=self.period:refresh='period'
            elif (stamp-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6:refresh='age'
        values=[];x=spatial
        for key,encoder,light,update,stride in (
            ('value',self.core.observation,None,self.core.fine_update,2),
            ('coarse',self.core.coarse_encoder,self.core.coarse_light,self.core.coarse_update,4),
            ('deep',self.core.deep_encoder,self.core.deep_light,self.core.deep_update,8)):
            current=(encoder if refresh or light is None else light)(x)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:
                bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride)
                old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            update.delta_t=dt;x=update(current,old,valid);values.append(x)
        features=self.core.readout(torch.cat([m(v) for m,v in zip(self.core.deblocks,values)],1))
        pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        new={k:v.detach() if detach_state else v for k,v in zip(self.state_keys,values)}
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,
            refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'],
            since_refresh=0 if refresh else state['since_refresh']+1)
        return pred,new,dict(reset=reason,full_refresh=refresh,
            cache_bytes=sum(new[k].numel()*new[k].element_size() for k in self.state_keys),
            middle_deep_encoder_executed=bool(refresh),state_updates_executed=True)

"""Readout-only recurrent fast path, with unchanged dense upstream states."""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_mechanism_probes import FastReadoutUpdate
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class ReadoutRefreshDetector(PyramidStateDetector):
    """Dense, learned correction or pure reuse, each with its own readout state.

    The expensive readout is never evaluated on fast ordinary frames. Upstream
    encoders and recurrent updates remain dense, isolating readout savings.
    Dense controls retain an unused fast module and readout state for accounting.
    """
    def __init__(self,base,mode='fast',period=4,max_age=.45):
        nn.Module.__init__(self)
        if type(base) is not PyramidStateDetector or base.memory_policy!='recurrent':raise ValueError('Recurrent base required')
        if mode not in ('dense','fast','reuse'):raise ValueError(mode)
        if type(period) is not int or not 1<=period<=6 or not 0<max_age<=.65:raise ValueError('Invalid refresh schedule')
        self.core=copy.deepcopy(base.core);self.core.fast_readout=FastReadoutUpdate()
        self.pillar_vfe,self.scatter=base.pillar_vfe,base.scatter
        self.bounds=copy.deepcopy(base.bounds);self.memory_policy='recurrent';self.spatial_only=False
        self.mode,self.period,self.max_age=mode,period,max_age
        self.variant='readout_refresh_'+mode;self.train(False)

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if spatial.shape[0]!=1 or requests is not None:raise ValueError('Single stream; no local masks')
        reason,_=reset_reason(state,meta);stamp=int(meta['timestamp_us']);refresh=reason
        if refresh is None:
            if state['since_refresh']+1>=self.period:refresh='period'
            elif (stamp-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6:refresh='age'
        values=[];x=spatial
        for key,encoder,update,stride in (
            ('value',self.core.observation,self.core.fine_update,2),
            ('coarse',self.core.coarse_encoder,self.core.coarse_update,4),
            ('deep',self.core.deep_encoder,self.core.deep_update,8)):
            current=encoder(x)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:
                bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride)
                old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            x=update(current,old,valid);values.append(x)
        joined=torch.cat([m(v) for m,v in zip(self.core.deblocks,values)],1)
        heavy=bool(refresh) or self.mode=='dense'
        holes=0.
        if heavy:features=self.core.readout(joined)
        else:
            bounds=block_lidar_range(self.bounds,spatial.shape[-2:],state['readout'].shape[-2:],2)
            old,valid=warp_bev(state['readout'],state['pose'],meta['pose'],bounds)
            holes=float((~valid).float().mean())
            features=self.core.fast_readout(joined,old,valid) if self.mode=='fast' else old
        pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        new={k:v.detach() if detach_state else v for k,v in zip(self.state_keys,values)}
        new['readout']=features.detach() if detach_state else features
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,
            since_refresh=0 if refresh else state['since_refresh']+1,
            refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        return pred,new,dict(reset=reason,full_refresh=refresh,readout_executed=heavy,holes_fraction=holes,
            readout_cache_bytes=features.numel()*features.element_size(),
            cache_bytes=sum(new[k].numel()*new[k].element_size() for k in (*self.state_keys,'readout')))

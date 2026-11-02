"""Map trained protected ANN state directly to membrane plus event communication.

No second retention filter or duplicate task history. Learned ANN gates and
protected refresh functions are unchanged. Only sent/projection are extra state.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector,reset_reason
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_event_projection import sparse_deconv_events,warp_projection_phases


class ProtectedEventDetector(ProtectedMultiscaleRefreshDetector):
    def __init__(self,trained,mode='continuous',quantum=.25):
        nn.Module.__init__(self)
        if type(trained) is not ProtectedMultiscaleRefreshDetector or mode not in ('continuous','quant_dense','event_phase') or quantum<=0:raise ValueError('Trained protected model and valid mode/quantum required')
        self.core=copy.deepcopy(trained.core);self.pillar_vfe,self.scatter=trained.pillar_vfe,trained.scatter
        self.bounds=copy.deepcopy(trained.bounds);self.memory_policy='recurrent';self.spatial_only=False
        self.period,self.max_age,self.light_observations=trained.period,trained.max_age,trained.light_observations
        self.event_mode,self.quantum=mode,quantum;self.variant='protected_'+mode;self.train(False)

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if torch.is_grad_enabled() or spatial.shape[0]!=1 or requests is not None:raise ValueError('Frozen single-stream event interface')
        reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us']);refresh=reason
        if refresh is None:
            if state['since_refresh']+1>=self.period:refresh='period'
            elif (stamp-state['refresh_timestamp_us'])/1e6>=self.max_age-1e-6:refresh='age'
        x=spatial;values=[];new={};events=None;event_macs=0;projection_out=None
        for key,encoder,light,update,stride in (
            ('value',self.core.observation,None,self.core.fine_update,2),
            ('coarse',self.core.coarse_encoder,self.core.coarse_light,self.core.coarse_update,4),
            ('deep',self.core.deep_encoder,self.core.deep_light,self.core.deep_update,8)):
            current=(encoder if refresh or light is None or not self.light_observations else light)(x)
            bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            if refresh and key in self.core.refresh_updates:update=self.core.refresh_updates[key]
            update.delta_t=dt;v=update(current,old,valid);new[key]=v;x=v
            if key=='coarse' and self.event_mode!='continuous':
                previous=torch.zeros_like(v) if reason else warp_bev(state['sent'],state['pose'],meta['pose'],bounds)[0]
                counts=torch.trunc((v-previous)/self.quantum);sent=previous+self.quantum*counts;new['sent']=sent;x=sent
                events=dict(active_elements=int(torch.count_nonzero(counts)),elements=counts.numel(),signed_unit_events=int(counts.abs().double().sum()),max_representation_error=float((v-sent).abs().max()))
                if self.event_mode=='event_phase':
                    conv=self.core.deblocks[1][0]
                    if refresh:z=conv(sent)
                    else:
                        aligned,zvalid=warp_projection_phases(state['projection'],state['pose'],meta['pose'],bounds);assert torch.equal(zvalid,valid)
                        delta,account=sparse_deconv_events(conv,self.quantum*counts);event_macs=account['weighted_event_macs'];z=aligned+delta
                    new['projection']=z;projection_out=self.core.deblocks[1][1:](z)
            values.append(x)
        features=self.core.readout(torch.cat([projection_out if i==1 and projection_out is not None else m(v) for i,(m,v) in enumerate(zip(self.core.deblocks,values))],1))
        pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        size=sum(v.numel()*v.element_size() for v in new.values())
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,since_refresh=0 if refresh else state['since_refresh']+1,refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        return pred,new,dict(reset=reason,full_refresh=refresh,cache_bytes=size,events=events,weighted_event_macs=event_macs,
            middle_deep_encoder_executed=bool(refresh) or not self.light_observations,state_updates_executed=True)

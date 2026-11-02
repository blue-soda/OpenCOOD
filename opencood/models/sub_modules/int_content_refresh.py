"""Content-driven refresh of trained middle/deep light paths, not a sparse SNN."""
import copy,math
import torch
from opencood.models.sub_modules.int_protected_events import ProtectedEventDetector
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


def decide_refresh(meta,state,evidence,mode='period4',threshold=1.):
    if mode not in ('period4','content') or not math.isfinite(evidence) or evidence<0 or not math.isfinite(threshold) or threshold<0:raise ValueError('Invalid causal refresh inputs')
    reason,_=reset_reason(state,meta)
    if reason:return reason
    if state['since_refresh']+1 >= (4 if mode=='period4' else 6):return 'period'
    if (int(meta['timestamp_us'])-state['refresh_timestamp_us'])/1e6 >= (.45 if mode=='period4' else .65)-1e-6:return 'age'
    return 'content' if mode=='content' and evidence>=threshold else None


def replay_content(rows,mode,threshold=1.):
    state=None;plan=[]
    for row in rows:
        reason=decide_refresh(row,state,row['evidence'],mode,threshold);stamp=int(row['timestamp_us'])
        state=dict(scene=row['segment'],timestamp_us=stamp,since_refresh=0 if reason else state['since_refresh']+1,refresh_timestamp_us=stamp if reason else state['refresh_timestamp_us'])
        plan.append(reason)
    return plan


class ContentRefreshDetector(ProtectedEventDetector):
    def __init__(self,trained,mode='period4',threshold=1.):
        super().__init__(trained,'continuous');self.policy_mode=mode;self.threshold=threshold

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if torch.is_grad_enabled() or requests is not None:raise ValueError('Frozen content policy engineering')
        reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us']);current=self.core.observation(spatial)
        if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
        else:old,valid=warp_bev(state['value'],state['pose'],meta['pose'],block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],2))
        fine=self.core.fine_update(current,old,valid)
        evidence=float((fine-old).abs().sum()/(fine.abs().sum()+old.abs().sum()+1e-6))
        refresh=decide_refresh(meta,state,evidence,self.policy_mode,self.threshold)
        values=[fine];x=fine
        for key,encoder,light,update,stride in (
            ('coarse',self.core.coarse_encoder,self.core.coarse_light,self.core.coarse_update,4),
            ('deep',self.core.deep_encoder,self.core.deep_light,self.core.deep_update,8)):
            current=(encoder if refresh or not self.light_observations else light)(x)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:old,valid=warp_bev(state[key],state['pose'],meta['pose'],block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride))
            if refresh:update=self.core.refresh_updates[key]
            update.delta_t=dt;x=update(current,old,valid);values.append(x)
        features=self.core.readout(torch.cat([m(v) for m,v in zip(self.core.deblocks,values)],1));pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        new={k:v for k,v in zip(self.state_keys,values)};size=sum(v.numel()*v.element_size() for v in values)
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,since_refresh=0 if refresh else state['since_refresh']+1,refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        return pred,new,dict(reset=reason,full_refresh=refresh,cache_bytes=size,evidence=evidence,age=new['since_refresh'],selector_elements=fine.numel(),middle_deep_encoder_executed=bool(refresh) or not self.light_observations)

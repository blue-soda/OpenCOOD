"""Frozen task membrane intervention with dense or exact phase-event readout.

Only the coarse recurrence is changed. The fine/deep paths remain ANN, as do
all encoders and the coarse candidate. This is not a full SNN detector.
"""
import copy
import torch
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,reset_reason
from opencood.models.sub_modules.int_task_membrane import TaskPotentialEvents
from opencood.models.sub_modules.int_event_projection import sparse_deconv_events,warp_projection_phases
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class TaskMembraneDetector(TemporalPyramidDetector):
    def __init__(self,base,mode,quantum=.25):
        if mode not in ('continuous','reset','events_dense','events_phase'):raise ValueError(mode)
        super().__init__(base,'leaky');self.task_mode=mode;self.quantum=quantum
        if mode.startswith('events'):
            self.core.coarse_update=TaskPotentialEvents(base.core.coarse_update,quantum,.2)
        self.train(False)

    def step_spatial(self,spatial,state,meta,detach_state=True,requests=None):
        if (torch.is_grad_enabled() and self.task_mode.startswith('events')) or requests is not None:raise ValueError('Events are inference-only; no spatial requests')
        reason,dt=reset_reason(state,meta);stamp=int(meta['timestamp_us']);refresh=reason
        if not refresh and self.task_mode=='events_phase':
            if state['projection_age']+1>=4:refresh='period'
            elif (stamp-state['projection_stamp'])/1e6>=.45-1e-6:refresh='age'
        x=spatial;values=[];new={};stats=None;event_macs=0;coarse_projection=None
        for key,encoder,update,stride in (
                ('value',self.core.observation,self.core.fine_update,2),
                ('coarse',self.core.coarse_encoder,self.core.coarse_update,4),
                ('deep',self.core.deep_encoder,self.core.deep_update,8)):
            current=encoder(x);bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],stride)
            reset=reason or (key=='coarse' and self.task_mode=='reset')
            if reset:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            if key=='coarse' and self.task_mode.startswith('events'):
                sent=torch.zeros_like(current) if reason else warp_bev(state['sent'],state['pose'],meta['pose'],bounds)[0]
                v,sent,counts,stats=update(current,old,sent,valid,dt);new[key]=v;new['sent']=sent;x=sent
                if self.task_mode=='events_phase':
                    conv=self.core.deblocks[1][0]
                    if refresh:projection=conv(sent)
                    else:
                        prior,pvalid=warp_projection_phases(state['projection'],state['pose'],meta['pose'],bounds)
                        assert torch.equal(valid,pvalid)
                        delta,account=sparse_deconv_events(conv,self.quantum*counts);event_macs=account['weighted_event_macs'];projection=prior+delta
                    new['projection']=projection;coarse_projection=self.core.deblocks[1][1:](projection)
            else:
                update.delta_t=0. if reset else dt;x=update(current,old,valid);new[key]=x
            values.append(x)
        ups=[coarse_projection if i==1 and coarse_projection is not None else m(v) for i,(m,v) in enumerate(zip(self.core.deblocks,values))]
        features=self.core.readout(torch.cat(ups,1));pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        size=sum(v.numel()*v.element_size() for v in new.values())
        if detach_state:new={k:v.detach() for k,v in new.items()}
        new.update(pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp,
            projection_age=0 if refresh else state['projection_age']+1,projection_stamp=stamp if refresh else state['projection_stamp'])
        return pred,new,dict(reset=reason,projection_refresh=refresh,cache_bytes=size,events=stats,weighted_event_macs=event_macs,
            boundary='Coarse membrane replaces ANN h; sent/projection are extra execution state. Fine/deep retain ANN recurrence. Sparse MACs need addition to Conv-hook FLOPs.')

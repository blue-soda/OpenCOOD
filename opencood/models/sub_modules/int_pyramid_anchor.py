"""Whole late-scale keyframes; anchor stays at its actual computation pose."""
import copy
import math
import torch
from torch import nn
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class AnchoredLatePyramidDetector(PyramidStateDetector):
    def __init__(self, base, period=2, max_age=.25, coverage='strict'):
        nn.Module.__init__(self)
        if type(base) is not PyramidStateDetector or base.memory_policy!='recurrent':
            raise ValueError('Ungated recurrent pyramid required')
        if type(period) is not int or not 1<=period<=3 or not math.isfinite(max_age) or not 0<max_age<=.25:
            raise ValueError('Invalid period/age')
        if coverage not in ('strict','zero'):raise ValueError('Invalid coverage policy')
        self.core,self.pillar_vfe,self.scatter=base.core,base.pillar_vfe,base.scatter
        self.bounds,self.memory_policy,self.variant=base.bounds,base.memory_policy,base.variant
        self.period,self.max_age,self.coverage=period,max_age,coverage
        self.train(False)

    def reuse_deep(self, current, old, valid):
        return old, None

    def observe_coarse(self, current, old, valid, reset):
        return None

    def decide_refresh(self, state, meta, reason, age, uncovered, count, evidence):
        refresh=reason
        if refresh is None:
            if count>=self.period:refresh='period'
            elif age>=self.max_age-1e-6:refresh='age'
            elif self.coverage=='strict' and uncovered>0:refresh='coverage'
        return refresh,None,None

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if self.training or torch.is_grad_enabled() or spatial.shape[0]!=1 or requests is not None:
            raise ValueError('Frozen single stream, no spatial masks')
        stamp=int(meta['timestamp_us']);reason='initial' if state is None else None
        if meta.get('reset_before',False):reason='explicit'
        if state is not None and reason is None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                gap=(stamp-state['timestamp_us'])/1e6
                if gap<=0:raise ValueError('Noncausal state')
                if gap>.25:reason='gap'
        x=spatial;values=[]
        for key,encoder,update,scale in [('value',self.core.observation,self.core.fine_update,2),
                ('coarse',self.core.coarse_encoder,self.core.coarse_update,4)]:
            current=encoder(x)
            if reason:old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:
                bounds=block_lidar_range(self.bounds,spatial.shape[-2:],current.shape[-2:],scale)
                old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
            x=update(current,old,valid);values.append(x)
        evidence=self.observe_coarse(x,old,valid,reason)
        shape=((x.shape[-2]+1)//2,(x.shape[-1]+1)//2)
        bounds=block_lidar_range(self.bounds,spatial.shape[-2:],shape,8)
        refresh=reason;age=0.;uncovered=0.;count=0
        if reason:
            old=x.new_zeros(1,self.core.deep_update.history.out_channels,*shape)
            valid=torch.zeros_like(old[:,:1],dtype=torch.bool)
        else:
            anchor=state['anchor'];age=(stamp-anchor['timestamp_us'])/1e6;count=anchor['skipped']+1
            if age<=0:raise ValueError('Noncausal anchor')
            old,valid=warp_bev(anchor['value'],anchor['pose'],meta['pose'],bounds)
            uncovered=float((~valid).float().mean())
        refresh,controller_state,controller_info=self.decide_refresh(state,meta,reason,age,uncovered,count,evidence)
        repair = None
        if refresh:
            deep=self.core.deep_update(self.core.deep_encoder(x),old,valid)
            anchor=dict(value=deep.detach(),pose=copy.deepcopy(meta['pose']),timestamp_us=stamp,skipped=0)
        else:
            deep,repair=self.reuse_deep(x,old,valid)
            anchor=dict(state['anchor'],skipped=count)
        values.append(deep)
        features=self.core.readout(torch.cat([layer(v) for layer,v in zip(self.core.deblocks,values)],1))
        pred=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:pred['dm']=self.core.dir_head(features)
        new=dict(value=values[0].detach(),coarse=values[1].detach(),anchor=anchor,
            pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp)
        if controller_state is not None:new['controller']=controller_state
        size=sum(v.numel()*v.element_size() for v in (new['value'],new['coarse'],anchor['value']))
        return pred,new,dict(reset=reason,late_refresh=refresh,repair=repair,anchor_timestamp_us=anchor['timestamp_us'],
            refresh_evidence=evidence,controller=controller_info,
            anchor_age_before_refresh=age,uncovered_fraction=uncovered,state_bytes=size,cache_bytes=size,
            cache_stages=[dict(actual_write_fraction=1.,executed_input_area_ratio=1.) for _ in range(2)]+
                         [dict(actual_write_fraction=repair['actual_write_fraction'],
                               executed_input_area_ratio=repair['executed_input_area_ratio']) if repair is not None else
                          dict(actual_write_fraction=float(bool(refresh)),executed_input_area_ratio=float(bool(refresh)))])

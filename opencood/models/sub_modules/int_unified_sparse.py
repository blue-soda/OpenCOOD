"""Inference-time selective state updates; evidence encoder and readout stay dense."""
import math
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import refresh_tiles, tile_mask, expand_tiles


def change_scores(current,reference,tile):
    pool=lambda x:F.avg_pool2d(x,tile,tile,ceil_mode=True,count_include_pad=False)
    error=pool((current-reference).square().mean(1,keepdim=True))
    energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),pool(reference.square().mean(1,keepdim=True)))
    return (error/energy.clamp_min(1e-6)).sqrt()


class SelectiveUnifiedBEV:
    """Reference evidence is updated only where the persistent state is updated.

    All three cached maps share the last frame's coordinates and are warped together.
    Selection never reads GT, teacher predictions, or a dense current state update.
    """
    def __init__(self,student,threshold,tile=16,max_age=.5,execution='sparse',fallback_fraction=.8):
        if student.memory_policy!='recurrent':raise ValueError('Selective updates require recurrent weights')
        if execution not in ('sparse','masked_dense','full'):raise ValueError('Unknown execution')
        if not math.isfinite(threshold) or threshold<0 or tile<1 or max_age<=0:raise ValueError('Invalid gate')
        self.student=student;self.threshold=threshold;self.tile=tile;self.max_age=max_age
        self.execution=execution;self.fallback_fraction=fallback_fraction
        self.last_scores=None;self.last_eligible=None

    def step(self,data,state,meta):
        return self.step_spatial(self.student.spatial(data),state,meta)

    def step_spatial(self,spatial,state,meta):
        if self.student.training:raise ValueError('Selective executor is inference-only')
        core=self.student.core;evidence=core.evidence(spatial)
        channels=core.channels;ec=evidence.shape[1]
        reason='initial' if state is None else None;gap=0.
        if meta.get('reset_before',False):reason='explicit'
        if state is not None and reason is None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                gap=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
                if gap<=0:raise ValueError('Non-causal selective state')
                if gap>.25:reason='gap'
        if reason is not None:
            old=evidence.new_zeros((1,channels,*evidence.shape[-2:]))
            reference=torch.zeros_like(evidence);age=torch.zeros_like(old[:,:1])
            valid=torch.zeros_like(age,dtype=torch.bool)
        else:
            packed_history=torch.cat([state['value'],state['reference'],state['age']],1)
            aligned,valid=warp_bev(packed_history,state['pose'],meta['pose'],self.student.bounds)
            old=aligned[:,:channels];reference=aligned[:,channels:channels+ec]
            age=aligned[:,-1:]+gap
        scores=change_scores(evidence,reference,self.tile)
        eligible=~tile_mask(~valid,self.tile)
        self.last_scores=scores.detach();self.last_eligible=eligible.detach()
        changed=expand_tiles(scores>self.threshold,self.tile,evidence.shape[-2:])
        requested=changed | ~valid | (age>=self.max_age)
        packed=torch.cat([evidence,old,valid.to(old.dtype)],1)
        if self.execution=='full' or reason is not None:
            value=core.update(packed);written=torch.ones_like(valid)
            stats=dict(actual_write_fraction=1.,executed_input_area_ratio=1.,fallback=False,tiles=1,
                       crop_area_ratio_before_fallback=1.,requested_fraction=1.,selected_fraction=1.)
        else:
            value,written,stats=refresh_tiles(core.update,packed,old,requested,1,core.update.radius,
                self.tile,self.fallback_fraction,execution=self.execution)
        reference=torch.where(written,evidence,reference)
        age=torch.where(written,torch.zeros_like(age),age)
        new=dict(value=value.detach(),reference=reference.detach(),age=age.detach(),pose=meta['pose'],
                 scene=meta['segment'],timestamp_us=int(meta['timestamp_us']))
        size=sum(new[k].numel()*new[k].element_size() for k in ['value','reference','age'])
        info=dict(reset=reason,cache_bytes=size,state_bytes=size,unified_dense_update=self.execution=='full',
                  original_backbone_called=False,update_execution=self.execution,update_stats=stats,
                  reused_fraction=1.-stats['actual_write_fraction'],max_cell_age=float(age.max()),
                  cache_stages=[dict(actual_write_fraction=0.,executed_input_area_ratio=0.) for _ in range(3)])
        return core.predict(value),new,info

"""Reuse expensive residual/gate outputs while updating the state every frame."""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles
from opencood.models.sub_modules.int_unified_gather import gathered_update
from opencood.models.sub_modules.int_unified_bev import ResidualStateUpdate


class CachedUpdateParts:
    """Frozen V2 experiment: periodic anchors and local new-view/age repair.

    Current evidence/current projection, recurrent value and readout are updated on
    every frame. Only the expensive net's residual and *logits* are cached.
    """
    def __init__(self,student,period=2,tile=16,max_age=.25,repair='gather'):
        if student.memory_policy!='recurrent' or not isinstance(student.core.update,ResidualStateUpdate):
            raise ValueError('Requires recurrent V2 weights')
        if period<1 or tile<1 or max_age<=0 or repair not in ('gather','masked_dense'):
            raise ValueError('Invalid cache policy')
        self.student=student;self.period=period;self.tile=tile;self.max_age=max_age;self.repair=repair

    def step(self,data,state,meta):
        return self.step_spatial(self.student.spatial(data),state,meta)

    def step_spatial(self,spatial,state,meta):
        if self.student.training or torch.is_grad_enabled():raise ValueError('Frozen inference only')
        core=self.student.core;module=core.update;evidence=core.evidence(spatial);c=core.channels
        reason='initial' if state is None else None;gap=0.
        if meta.get('reset_before',False):reason='explicit'
        if state is not None and reason is None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                gap=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
                if gap<=0:raise ValueError('Non-causal cached parts')
                if gap>.25:reason='gap'
        if reason is not None:
            old=evidence.new_zeros((1,c,*evidence.shape[-2:]));parts=torch.cat([old,old],1)
            age=torch.zeros_like(old[:,:1]);valid=torch.zeros_like(age,dtype=torch.bool);since=0
        else:
            aligned,valid=warp_bev(torch.cat([state['value'],state['parts'],state['age']],1),
                                   state['pose'],meta['pose'],self.student.bounds)
            old=aligned[:,:c];parts=aligned[:,c:3*c];age=aligned[:,-1:]+gap
            since=state['since_anchor']+1
        packed=torch.cat([evidence,old,valid.to(old.dtype)],1)
        anchor=reason is not None or since>=self.period
        fallback=False
        if anchor:
            parts=module.net(packed);written=torch.ones_like(valid);since=0;area=1.
        else:
            written=expand_tiles(tile_mask(~valid | (age>=self.max_age),self.tile),self.tile,old.shape[-2:])
            support=F.max_pool2d(written.float(),5,1,2)
            area=float(support.mean());fallback=area>=.8
            if fallback:
                parts=module.net(packed);written=torch.ones_like(valid);area=1.;since=0
            elif self.repair=='masked_dense':
                parts=torch.where(written,module.net(packed),parts);area=1.
            else:
                parts=gathered_update(module,packed,written,cached_parts=parts)
        residual,gate=parts.chunk(2,dim=1)
        retain=gate.sigmoid()*valid.to(old.dtype)
        value=retain*old+(1-retain)*(module.current(evidence)+residual)
        age=torch.where(written,torch.zeros_like(age),age)
        new=dict(value=value.detach(),parts=parts.detach(),age=age.detach(),pose=meta['pose'],
                 scene=meta['segment'],timestamp_us=int(meta['timestamp_us']),since_anchor=since)
        size=sum(new[k].numel()*new[k].element_size() for k in ['value','parts','age'])
        info=dict(reset=reason,cache_bytes=size,state_bytes=size,original_backbone_called=False,
                  parts_anchor=anchor,parts_fallback=fallback,parts_write_fraction=float(written.float().mean()),
                  parts_input_support_fraction=area,max_parts_age=float(age.max()),state_updated_everywhere=True,
                  cache_stages=[dict(actual_write_fraction=0.,executed_input_area_ratio=0.) for _ in range(3)])
        return core.predict(value),new,info

"""Short update path with complete D writeback, separate from fixed anchors."""
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_anchor_residual import predict


class RecurrentDeepState:
    def __init__(self,repair,bounds,period=6):
        if period not in (1,3,4,6):raise ValueError('Period1/3/4/6 required')
        self.repair,self.bounds,self.period=repair,bounds,period
        self.max_age_us={1:250000,3:250000,4:350000,6:550000}[period]

    def update(self,h,old,valid,update_mask=None):
        return self.repair(torch.cat((h,old,valid.to(old.dtype)),1)),dict(deep_write_fraction=1.,deep_short_macs_ratio=1.)

    def step(self,h,state,meta,full_value,reset=False,detach=True,update_mask=None):
        stamp=int(meta['timestamp_us']);reason=None;detail=dict(deep_write_fraction=1.)
        if state is None or reset or state['scene']!=meta['segment']:reason='reset'
        else:
            if stamp<=state['stamp']:raise ValueError('Noncausal D state')
            if stamp-state['full_stamp']>=self.max_age_us:reason='age'
            elif state['frames']+1>=self.period:reason='period'
        if reason:
            value=full_value();frames=0;full_stamp=stamp;uncovered=0.
        else:
            old,valid=warp_bev(state['value'],state['pose'],meta['pose'],self.bounds)
            value,detail=self.update(h,old,valid,update_mask)
            frames=state['frames']+1;full_stamp=state['full_stamp'];uncovered=float((~valid).float().mean())
        new=dict(value=value.detach() if detach else value,pose=meta['pose'],stamp=stamp,
                 full_stamp=full_stamp,frames=frames,scene=meta['segment'])
        info=dict(deep_partial=reason is None,deep_refresh_reason=reason,
            deep_age_s=(stamp-full_stamp)/1e6,deep_depth=frames,deep_uncovered_fraction=uncovered,
            anchor_bytes=value.numel()*value.element_size())
        info.update(detail);return value,new,info


class RecurrentDeepDetector:
    """Frozen mature observation/readout; train only recursive complete-D repair.

    Supervised mode calculates a discarded dense teacher on the student's own
    current H. That target enters state only on declared true refresh frames.
    Inference never evaluates this teacher on ordinary update frames.
    """
    def __init__(self,baseline,repair,period=6):
        if baseline.readout_cache is not None:raise ValueError('Isolate D update before composing readout cache')
        self.model,self.readout,self.spatial=baseline.model,baseline.readout,baseline.spatial
        self.repair=repair;self.bounds=baseline.bounds
        self.deep=RecurrentDeepState(repair,self.bounds,period)
        self.readout_cache=None

    @torch.no_grad()
    def full_deep(self,h):
        m=self.model;second=m.backbone.resnet.layer1(h);third=m.backbone.resnet.layer2(second)
        return self.readout.history(torch.cat((m.backbone.deblocks[1](second),m.backbone.deblocks[2](third)),1))

    def step(self,data,state,meta,supervise=False):
        if supervise and self.readout_cache is not None:raise ValueError('Readout reuse is frozen inference only')
        m=self.model;p=data['processed_lidar']
        with torch.no_grad():
            if p['voxel_features'].shape[0]:
                batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')};batch['batch_size']=1
                spatial=m.scatter(m.pillar_vfe(batch))['spatial_features']
            else:
                nx,ny,_=m.scatter.model_cfg['grid_size'];spatial=m.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
            temporal,fm,info=m.feature_memory.step(spatial,None if state is None else state['fm'],meta)
            h,spatial_state,detail=self.spatial.step(temporal,None if state is None else state['spatial'],meta,bool(info['reset']))
            current=self.readout.current(m.backbone.deblocks[0](h))
            target=self.full_deep(h) if supervise else None
            target_pred=predict(m,self.readout.fuse(current,target)) if supervise else None
        value,deep,deep_info=self.deep.step(h,None if state is None else state['deep'],meta,
            lambda:target if target is not None else self.full_deep(h),bool(info['reset']),detach=not supervise,
            update_mask=detail.get('spatial_update_mask'))
        readout_state=None
        if self.readout_cache is None:
            features=self.readout.fuse(current,value)
        else:
            features,readout_state,readout_info=self.readout_cache.step(current+value,
                None if state is None else state['readout'],meta,bool(info['reset']))
            if not readout_info['readout_refresh'] and self.readout_cache.execution=='selected':
                cache=self.readout_cache
                full=current.shape[-2]*current.shape[-1]*sum(c.weight.numel() for c in (cache.first,cache.project,cache.second))
                readout_info['readout_macs_saved']=full-readout_info['readout_extra_linear_macs']
            detail.update(readout_info)
        pred=predict(m,features)
        confidence=pred['psm'].detach().sigmoid().amax(1,keepdim=True)
        if readout_state is not None:
            assert confidence.shape[-2:]==features.shape[-2:]
            if hasattr(self.readout_cache,'set_confidence'):self.readout_cache.set_confidence(readout_state,confidence)
            else:readout_state['confidence']=confidence
            detail['readout_state_bytes']+=confidence.numel()*confidence.element_size()
        if confidence.shape[-2:]!=h.shape[-2:]:
            confidence=torch.nn.functional.interpolate(confidence,size=h.shape[-2:],mode='nearest')
        spatial_state['confidence']=confidence
        detail['spatial_state_bytes']+=confidence.numel()*confidence.element_size()
        info.update(detail);info.update(deep_info)
        if supervise:info.update(target_deep=target,target_predictions=target_pred,student_deep=value)
        new=dict(fm=fm,spatial=spatial_state,deep=deep)
        if readout_state is not None:new['readout']=readout_state
        return pred,new,info

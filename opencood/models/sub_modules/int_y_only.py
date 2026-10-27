"""One complete spatial Y state; ordinary frames never execute H or D.

The frozen low-rank GRU retains its own FM state. Y-only refers to spatial
caching, not to the number of recurrent tensors in the entire detector.
"""
import torch
from torch import nn
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_anchor_residual import predict


class YOnlyUpdater(nn.Module):
    """Five-by-five dependency; bounded direct old-Y contribution."""
    def __init__(self, current_channels=128, channels=256, width=64):
        super().__init__()
        self.current_channels,self.channels,self.width=current_channels,channels,width
        self.features=nn.Sequential(
            nn.Conv2d(current_channels+channels+1,width,1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1),nn.ReLU())
        self.candidate=nn.Conv2d(width,channels,1)
        self.gate=nn.Conv2d(width,1,1)
        nn.init.normal_(self.candidate.weight,std=.001)
        nn.init.zeros_(self.candidate.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,4.)

    def forward(self,evidence,old,valid):
        hidden=self.features(torch.cat((evidence,old,valid.to(old.dtype)),1))
        fresh=torch.relu(self.candidate(hidden))
        weight=.95*torch.sigmoid(self.gate(hidden))*valid.to(old.dtype)
        return weight*old+(1.-weight)*fresh


class YOnlyState:
    def __init__(self,updater,bounds,period=3):
        if period not in (1,3,5):raise ValueError('Y refresh period must be 1, 3 or 5')
        self.updater,self.bounds,self.period=updater,bounds,period
        self.max_age_us=450000 if period==5 else 250000

    def reason(self,state,meta,reset=False):
        if state is None or reset or state['scene']!=meta['segment']:return 'reset'
        stamp=int(meta['timestamp_us'])
        if stamp<=state['stamp']:raise ValueError('Noncausal Y state')
        if stamp-state['full_stamp']>=self.max_age_us:return 'age'
        if state['frames']+1>=self.period:return 'period'
        return None

    def step(self,evidence,state,meta,full_value,reset=False,supervise=False):
        reason=self.reason(state,meta,reset)
        with torch.no_grad():target=full_value() if reason or supervise else None
        if reason:
            value=target;frames=0;full_stamp=int(meta['timestamp_us']);uncovered=0.
        else:
            old,valid=warp_bev(state['value'],state['pose'],meta['pose'],self.bounds)
            value=self.updater(evidence,old,valid)
            frames=state['frames']+1;full_stamp=state['full_stamp']
            uncovered=float((~valid).float().mean())
        new=dict(value=value if supervise else value.detach(),pose=meta['pose'],
                 stamp=int(meta['timestamp_us']),scene=meta['segment'],frames=frames,full_stamp=full_stamp)
        info=dict(y_refresh=reason is not None,y_refresh_reason=reason,y_depth=frames,y_uncovered_fraction=uncovered)
        if supervise:info.update(y_target=target,y_student=value)
        return value,new,info


class YOnlyDetector:
    def __init__(self,baseline,updater,period=3):
        self.model,self.readout,self.stage=baseline.model,baseline.readout,baseline.spatial.stage
        self.updater=updater
        self.cache=YOnlyState(updater,baseline.bounds,period)
        self.execution_costs={}
        self.readout_cache=None

    @torch.no_grad()
    def front(self,data,state,meta):
        m=self.model;p=data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')};batch['batch_size']=1
            spatial=m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx,ny,_=m.scatter.model_cfg['grid_size']
            spatial=m.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
        temporal,fm,info=m.feature_memory.step(spatial,None if state is None else state['fm'],meta)
        return self.stage.observe(temporal),fm,info

    @torch.no_grad()
    def full_from_evidence(self,evidence):
        # H/D are temporary current intermediates, never persistent states.
        h,_=self.stage.dense_from_stem(evidence);m=self.model
        second=m.backbone.resnet.layer1(h);third=m.backbone.resnet.layer2(second)
        shallow=self.readout.current(m.backbone.deblocks[0](h))
        deep=self.readout.history(torch.cat((m.backbone.deblocks[1](second),m.backbone.deblocks[2](third)),1))
        return self.readout.fuse(shallow,deep)

    def step(self,data,state,meta,supervise=False):
        if torch.is_grad_enabled() and not supervise:raise ValueError('Gradients require explicit supervision')
        evidence,fm,info=self.front(data,state,meta)
        value,y,details=self.cache.step(evidence,None if state is None else state['y'],meta,
            lambda:self.full_from_evidence(evidence),bool(info['reset']),supervise)
        pred=predict(self.model,value);refresh=details['y_refresh'];info.update(details)
        # Compatibility with paired evaluation. Real-data hook calibration
        # supplies whole-path MACs, rather than subtracting presumed savings.
        info.update(whole_model_macs=self.execution_costs.get(refresh),deep_partial=not refresh,
            spatial_refresh=refresh,spatial_extra_linear_macs=0,spatial_state_bytes=0,
            anchor_bytes=0,readout_state_bytes=value.numel()*value.element_size(),
            readout_refresh=refresh,readout_write_fraction=1.)
        return pred,dict(fm=fm,y=y),info

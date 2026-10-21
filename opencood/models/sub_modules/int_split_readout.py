"""Separate fresh shallow evidence from cacheable deep readout contributions.

The split is before any spatial convolution, BN or activation in shrink.
Only the bias-free deep projection is eligible for a future temporal cache.
This module alone does not implement temporal reuse or regional execution.
"""
import copy
import torch
from torch import nn
from opencood.models.sub_modules.int_anchor_residual import refresh_reason, predict
from opencood.models.sub_modules.int_feature_memory import warp_bev


class SplitCompactReadout(nn.Module):
    def __init__(self, shrink, current_channels):
        super().__init__()
        if len(shrink.layers) != 1:
            raise ValueError('One compact DoubleConv required')
        sequence = shrink.layers[0].double_conv
        if not isinstance(sequence[0], nn.Sequential) or len(sequence[0]) != 2:
            raise ValueError('Expected projection followed by spatial convolution')
        projection, spatial = sequence[0]
        if not isinstance(projection, nn.Conv2d) or projection.kernel_size != (1,1):
            raise ValueError('Expected a 1x1 projection')
        if projection.bias is not None or projection.groups != 1 or projection.stride != (1,1):
            raise ValueError('Only bias-free ordinary projections can be split')
        if projection.padding != (0,0) or not 0 < current_channels < projection.in_channels:
            raise ValueError('Invalid channel split or projection padding')
        self.current_channels = current_channels
        self.current = nn.Conv2d(current_channels, projection.out_channels, 1, bias=False)
        self.history = nn.Conv2d(projection.in_channels-current_channels, projection.out_channels, 1, bias=False)
        self.current.to(device=projection.weight.device, dtype=projection.weight.dtype)
        self.history.to(device=projection.weight.device, dtype=projection.weight.dtype)
        with torch.no_grad():
            self.current.weight.copy_(projection.weight[:,:current_channels])
            self.history.weight.copy_(projection.weight[:,current_channels:])
        self.spatial = copy.deepcopy(spatial)
        self.tail = copy.deepcopy(nn.Sequential(*list(sequence.children())[1:]))
        self.train(shrink.training)

    def fuse(self, current_projection, historical_projection):
        # Current evidence remains dense; neither BN nor spatial readout is cached.
        return self.tail(self.spatial(current_projection+historical_projection))

    def forward(self, decoded):
        return self.fuse(self.current(decoded[:,:self.current_channels]),
                         self.history(decoded[:,self.current_channels:]))


class SplitReadoutDetector:
    """Cache only the deep contribution before the spatial readout.

    Current stage1, its upsampling/projection, and all spatial readout layers
    run every frame. No intermediate prediction is written back to the anchor.
    This zero-correction diagnostic is not regional execution.
    """
    def __init__(self, model, config, readout, period=3, repair=None):
        if model.training or readout.training or period not in (1,3):
            raise ValueError('Frozen models and period1/3 required')
        if len(model.backbone.deblocks)!=3 or not model.shrink_flag:
            raise ValueError('Expected three direct multiscale branches and shrink')
        if model.backbone.deblocks[0][0].out_channels!=readout.current_channels:
            raise ValueError('Current branch channel split does not match')
        self.model,self.readout,self.period=model,readout,period
        self.repair=repair
        self.bounds=config['model']['args']['lidar_range']

    def step(self,data,state,meta):
        m=self.model;p=data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')}
            batch['batch_size']=1
            spatial=m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx,ny,_=m.scatter.model_cfg['grid_size']
            spatial=m.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
        temporal,fm,info=m.feature_memory.step(spatial,None if state is None else state['fm'],meta)
        first=m.backbone.resnet.layer0(temporal)
        current=self.readout.current(m.backbone.deblocks[0](first))
        cache=None if state is None else state['cache']
        reason=refresh_reason(cache,meta,info['reset'],self.period)
        if reason:
            second=m.backbone.resnet.layer1(first)
            third=m.backbone.resnet.layer2(second)
            deep=torch.cat([m.backbone.deblocks[1](second),m.backbone.deblocks[2](third)],1)
            projected=self.readout.history(deep)
            cache=dict(value=projected,pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
            uncovered=0.
        else:
            projected,valid=warp_bev(cache['value'],cache['pose'],meta['pose'],self.bounds)
            uncovered=float((~valid).float().mean())
            if self.repair is not None:
                projected=self.repair(torch.cat([first,projected,valid.to(projected.dtype)],1))
            cache=dict(cache,frames=cache['frames']+1)
        assert current.shape==projected.shape
        prediction=predict(m,self.readout.fuse(current,projected))
        info.update(local_partial=reason is None,anchor_stamp=cache['stamp'],
                    anchor_depth=cache['frames'],uncovered_fraction=uncovered,
                    anchor_bytes=cache['value'].numel()*cache['value'].element_size())
        return prediction,dict(fm=fm,cache=cache),info

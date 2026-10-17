"""Dependency-safe request masks for local repair; execution remains separate."""
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_feature_memory import warp_bev


def affected_outputs(changed_input, stride, radius):
    """Outputs whose receptive field touches a changed input position."""
    return F.max_pool2d(changed_input.float(), 2*radius+1, stride, radius) > 0


def local_change_mask(current, aligned, valid, threshold):
    pool = lambda x: F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
    error = pool((current-aligned).square().mean(1,keepdim=True))
    energy = torch.maximum(pool(current.square().mean(1,keepdim=True)),
                           pool(aligned.square().mean(1,keepdim=True)))
    scores = (error/energy.clamp_min(1e-6)).sqrt()
    return expand_tiles(scores > threshold,4,current.shape[-2:]) | ~valid


def local_requests(changed_stage1, valid_stage2, valid_stage3, tile):
    # Four stage2 BasicBlocks: radius15 in stage1 input. Five stage3: radius19.
    request2 = affected_outputs(changed_stage1,2,15) | ~valid_stage2
    written2 = expand_tiles(tile_mask(request2,tile),tile,request2.shape[-2:])
    # Refresh all stage3 outputs potentially affected by stage2 repair, including
    # boundary repairs. Just resizing the original change mask misses this halo.
    request3 = affected_outputs(written2,2,19) | ~valid_stage3
    written3 = expand_tiles(tile_mask(request3,tile),tile,request3.shape[-2:])
    return written2,written3


class LocalCacheDetector:
    """Actual frozen-stage crop execution, one partial frame per full refresh.

Keep the full-frame anchor until the next mandatory full refresh. Do not blend
different pose/time anchors into the saved reference. Stage3 requests use the
ACTUAL stage2 write mask, including a possible full-stage fallback.
"""
    def __init__(self,model,config,threshold,period=2,verify=False,repair_modules=None,repair_dense=False):
        if model.training or model.feature_memory.mode!='gru' or period not in (1,2):
            raise ValueError('Frozen GRU and period 1/2 required')
        self.model=model;self.bounds=config['model']['args']['lidar_range']
        assert config['model']['args']['base_bev_backbone']['layer_nums']==[3,4,5]
        self.threshold=threshold;self.period=period;self.verify=verify
        self.repair_modules=repair_modules
        self.repair_dense=repair_dense
        if repair_dense and repair_modules is None:raise ValueError('Dense repair requires small repair modules')

    def step(self,data,state,meta):
        m=self.model;p=data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch={k:p[k] for k in ['voxel_features','voxel_coords','voxel_num_points']}
            batch['batch_size']=1
            spatial=m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx,ny,_=m.scatter.model_cfg['grid_size']
            spatial=m.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
        temporal,fm,info=m.feature_memory.step(spatial,None if state is None else state['fm'],meta)
        first=m.backbone.resnet.layer0(temporal)
        desc=None if self.repair_dense else early_descriptor(first,4)
        cache=None if state is None or info['reset'] else state['cache']
        count=0 if cache is None else cache['frames']+1
        reason='initial' if cache is None else 'period' if count>=self.period else None
        if cache is not None:
            age=(int(meta['timestamp_us'])-cache['stamp'])/1e6
            if age<=0:raise ValueError('Non-causal local cache')
            if age>=.25:reason='age'
        stages=[first];stats=[dict(actual_write_fraction=1.,executed_input_area_ratio=1.)]
        boundary=[0.,0.];errors=[];changed_fraction=0.
        if reason is not None:
            x=first
            for i in [1,2]:
                x=getattr(m.backbone.resnet,'layer%d'%i)(x);stages.append(x)
                stats.append(dict(requested_fraction=1.,selected_fraction=1.,
                    crop_area_ratio_before_fallback=1.,tiles=1,fallback=False,
                    executed_input_area_ratio=1.,actual_write_fraction=1.))
            cache=dict(values=stages[1:],desc=desc,pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
        else:
            if self.repair_dense:
                changed=torch.zeros_like(first[:,:1],dtype=torch.bool)
            else:
                aligned,valid=warp_bev(cache['desc'],cache['pose'],meta['pose'],self.bounds)
                changed=expand_tiles(local_change_mask(desc,aligned,valid,self.threshold),4,first.shape[-2:])
            changed_fraction=float(changed.float().mean())
            x=first;written=None
            for index,radius in [(1,15),(2,19)]:
                old,valid=warp_bev(cache['values'][index-1],cache['pose'],meta['pose'],self.bounds)
                boundary[index-1]=float((~valid).float().mean())
                if self.repair_modules is None:
                    request=affected_outputs(changed if index==1 else written,2,radius) | ~valid
                    module=getattr(m.backbone.resnet,'layer%d'%index)
                    stage_input=x
                    result,written,stat=refresh_tiles(module,stage_input,old,request,2,radius,tile=4)
                else:
                    from opencood.models.sub_modules.int_light_repair import repair_input_request
                    stage_input,request=repair_input_request(first,old,valid,changed,index)
                    module=self.repair_modules[index-1]
                    # Each repair depends directly on current stage1 and its own
                    # historical feature, not on the other repair's output.
                    if self.repair_dense:
                        result=module(stage_input);written=torch.ones_like(valid)
                        stat=dict(requested_fraction=1.,selected_fraction=1.,crop_area_ratio_before_fallback=1.,
                                  tiles=1,fallback=False,executed_input_area_ratio=1.,actual_write_fraction=1.)
                    else:
                        result,written,stat=refresh_tiles(module,stage_input,old,request,1,1,tile=4,allow_fallback=False)
                if self.verify:
                    # Engineering only; never enabled in measured inference.
                    reference=module(stage_input)
                    error=(result-reference).abs().masked_select(written.expand_as(result))
                    maximum=float(error.max()) if error.numel() else 0.
                    assert torch.allclose(result[written.expand_as(result)],reference[written.expand_as(reference)],atol=1e-4,rtol=1e-4)
                    errors.append(maximum)
                x=result;stages.append(x);stats.append(stat)
            cache=dict(cache,frames=count)
        features=m.backbone.decode_multiscale_feature(stages)
        if m.shrink_flag:features=m.shrink_conv(features)
        pred=dict(psm=m.cls_head(features),rm=m.reg_head(features))
        if m.use_dir:pred['dm']=m.dir_head(features)
        tensors=cache['values']+[fm['value']]
        if cache['desc'] is not None:tensors.append(cache['desc'])
        info.update(cache_stages=stats,cache_bytes=sum(t.numel()*t.element_size() for t in tensors),
            local_refresh_reason=reason,local_partial=reason is None,local_changed_fraction=changed_fraction,
            local_boundary_fractions=boundary,local_verification_errors=errors)
        return pred,dict(fm=fm,cache=cache),info

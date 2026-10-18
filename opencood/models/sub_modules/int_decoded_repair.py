"""Current stage1 corrects a warped detection-head feature from a full anchor."""
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev


class DecodedCacheDetector:
    def __init__(self,model,config,threshold,period=3,verify=False,repair_modules=None,repair_dense=True):
        if model.training or model.feature_memory.mode!='gru' or period not in (1,2,3):
            raise ValueError('Frozen GRU and bounded refresh period required')
        if not repair_dense or repair_modules is None or len(repair_modules)!=1:
            raise ValueError('Decoded repair uses one dense residual module')
        self.model=model;self.period=period;self.verify=verify;self.repairs=repair_modules
        self.bounds=config['model']['args']['lidar_range']
        self.repair_target='decoded'

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
        cache=None if state is None or info['reset'] else state['cache']
        count=0 if cache is None else cache['frames']+1
        reason='initial' if cache is None else 'period' if count>=self.period else None
        if cache is not None:
            age=(int(meta['timestamp_us'])-cache['stamp'])/1e6
            if age<=0:raise ValueError('Non-causal decoded cache')
            if age>=.25:reason='age'
        uncovered=0.
        if reason is not None:
            second=m.backbone.resnet.layer1(first)
            third=m.backbone.resnet.layer2(second)
            features=m.backbone.decode_multiscale_feature([first,second,third])
            if m.shrink_flag:features=m.shrink_conv(features)
            assert features.shape[1]==256 and first.shape[1]==128
            assert features.shape[-2:]==first.shape[-2:]
            cache=dict(values=[features],pose=meta['pose'],stamp=int(meta['timestamp_us']),frames=0)
        else:
            old,valid=warp_bev(cache['values'][0],cache['pose'],meta['pose'],self.bounds)
            features=self.repairs[0](torch.cat([first,old,valid.to(old.dtype)],1))
            uncovered=float((~valid).float().mean())
            cache=dict(cache,frames=count)
        pred=dict(psm=m.cls_head(features),rm=m.reg_head(features))
        if m.use_dir:pred['dm']=m.dir_head(features)
        deep=0. if reason is None else 1.
        info.update(cache_stages=[dict(actual_write_fraction=v,executed_input_area_ratio=v) for v in [1.,deep,deep]],
            cache_bytes=sum(t.numel()*t.element_size() for t in cache['values']+[fm['value']]),
            local_partial=reason is None,local_refresh_reason=reason,local_verification_errors=[],
            cache_anchor_stamp=cache['stamp'],cache_age_seconds=(int(meta['timestamp_us'])-cache['stamp'])/1e6,
            cache_frames_since_refresh=cache['frames'],decoded_uncovered_fraction=uncovered,
            decoded_repair=reason is None)
        return pred,dict(fm=fm,cache=cache),info

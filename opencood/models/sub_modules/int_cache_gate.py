"""Small content descriptor and causal whole-stage refresh without a teacher."""
import math
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev

DESCRIPTOR_VERSION='groupmean16_avgpool2_block4_relative_rms_p90_v1'


def cache_descriptor(stage_input):
    b,c,h,w=stage_input.shape
    assert c%16==0
    return F.avg_pool2d(stage_input.reshape(b,16,c//16,h,w).mean(2),2,2,ceil_mode=True,
                        count_include_pad=False)


def descriptor_change(current, aligned):
    pool=lambda t:F.avg_pool2d(t,4,4,ceil_mode=True,count_include_pad=False)
    error=pool((current-aligned).square().mean(1,keepdim=True))
    energy=torch.maximum(pool(current.square().mean(1,keepdim=True)),
                         pool(aligned.square().mean(1,keepdim=True)))
    scores=(error/energy.clamp_min(1e-6)).sqrt().flatten()
    return float(scores.kthvalue(int(math.ceil(.9*scores.numel()))).values)


def content_stage_step(module,x,cache,pose,timestamp_us,lidar_range,period,max_age,threshold):
    if module.training or period<1 or max_age<=0 or not math.isfinite(threshold) or threshold<0:
        raise ValueError('Invalid content-cache policy')
    descriptor=cache_descriptor(x)
    score=None;count=0 if cache is None else cache['frames']+1
    reason='initial' if cache is None else None
    if cache is not None:
        age=(int(timestamp_us)-cache['timestamp_us'])/1e6
        if age<=0:raise ValueError('Non-causal content cache')
        if count>=period:reason='period'
        elif age>=max_age:reason='age'
        else:
            aligned,valid=warp_bev(cache['descriptor'],cache['pose'],pose,lidar_range)
            if not bool(valid.all()):reason='coverage'
            else:
                score=descriptor_change(descriptor,aligned)
                if score>threshold:reason='change'
    if reason is not None:
        output=module(x)
        assert descriptor.shape[-2:]==output.shape[-2:]
        new=dict(value=output,descriptor=descriptor,pose=pose,timestamp_us=int(timestamp_us),frames=0)
    else:
        # Only accepted reuse warps the large cache. The module is never called.
        output,_=warp_bev(cache['value'],cache['pose'],pose,lidar_range)
        new=dict(cache,frames=count)
    return output,new,reason,score

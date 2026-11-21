"""Check sparse refresh of the exact integer accumulator at the two upsampling heads."""
import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection


def refresh_projection(projector,current,previous,area,old_pose,pose,sample_offset,check_all):
    channels,fanout=projector.weight.shape
    levels,reference=current.split((channels,fanout),1)
    if previous is None:
        before=torch.zeros_like(levels)
        aligned=None
    else:
        # Warp only exact grid indices; previous integer sums never pass through FP32.
        _,_,(ids,missing)=DeepPhaseProjection.align_task(projector,levels[:,:1].float(),old_pose,pose,area)
        aligned=previous.flatten(2).index_select(2,ids).view_as(current).masked_fill(missing,0)
        before=aligned[:,:channels]
    delta=levels-before
    static_count=(levels!=0).sum(1,keepdim=True)
    delta_count=(delta!=0).sum(1,keepdim=True)
    reuse=delta_count<static_count
    changes=torch.where(reuse,delta,levels)
    spatial=levels.shape[-1]*levels.shape[-2]
    sample=torch.arange(0 if check_all else sample_offset%32,spatial,1 if check_all else 32,device=levels.device)
    if aligned is None:
        accumulator=reference.new_zeros(1,fanout,1,len(sample))
    else:
        old_sum=aligned[:,channels:].flatten(2)[:,:,sample]
        accumulator=old_sum.masked_fill(~reuse.flatten(2)[:,:,sample],0).unsqueeze(2).contiguous()
    code=changes.flatten(2)[:,:,sample].unsqueeze(2)
    projector.event_linear(code,accumulator)
    assert torch.equal(accumulator.flatten(2),reference.flatten(2)[:,:,sample])
    events=int(torch.count_nonzero(changes))
    return dict(sites=spatial,verified=len(sample),full=check_all,incremental_sites=int(reuse.sum()),
                dense_mac=channels*fanout*spatial,static_mac=int(static_count.sum())*fanout,
                delta_mac=int(delta_count.sum())*fanout,selected_mac=events*fanout,
                unit_ac=int(changes.abs().sum())*fanout,channels=channels,fanout=fanout,
                packed_state_bytes=current.numel()*current.element_size())


class RefreshProjectionAudit:
    def __init__(self):
        self.full_classes=set()

    def measure(self,model,state,previous,reason,meta,shape):
        temporal=reason=='period'
        full=temporal not in self.full_classes
        self.full_classes.add(temporal)
        out={}
        for name,stride,projector in [('coarse',4,model.coarse_projector),('deep',8,model.deep_projector)]:
            key=name+'_projection'
            packed=state[key]
            area=block_lidar_range(model.source.bounds,shape,packed.shape[-2:],stride)
            prior=previous[key] if temporal else None
            out[name]=refresh_projection(projector,packed,prior,area,previous['pose'] if temporal else None,
                                         meta['pose'],int(meta['frame']),full)
        return out

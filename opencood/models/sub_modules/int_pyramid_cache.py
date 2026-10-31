"""Cache complete pyramid states and crop encoder plus update computation.

Pillar, selectors, warps, full multiscale decoder and heads remain dense.
Partial results match the same-mask dense reference, not ordinary dense inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_local_state_cache import change_request


class PackedPyramidStage(nn.Module):
    def __init__(self, encoder, update):
        super().__init__()
        self.encoder, self.update = encoder, update
        self.input_channels = encoder[0].conv1.in_channels
        self.output_channels = encoder[-1].conv2.out_channels
        # First block radius3, each following block adds4 input pixels;
        # two update 3x3 convolutions add another4 at output stride2.
        self.radius = 4 * len(encoder) + 3

    def pack(self, current, old, valid):
        history = torch.cat((old, valid.to(old.dtype)), 1)
        history = history.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return torch.cat((current, history[..., :current.shape[-2], :current.shape[-1]]), 1)

    def forward(self, packed):
        current = self.encoder(packed[:, :self.input_channels])
        history = packed[:, self.input_channels:, ::2, ::2]
        return self.update(current, history[:, :self.output_channels], history[:, -1:] > 0)


class CachedPyramidDetector(PyramidStateDetector):
    """Inference adapter sharing frozen base weights, with per-cell age guards."""
    def __init__(self, base, fraction=.25, tile=8, max_cache_age=.5,
                 execution='sparse', allow_fallback=True):
        nn.Module.__init__(self)
        if type(base) is not PyramidStateDetector or base.memory_policy != 'recurrent':
            raise ValueError('Only the ungated recurrent pyramid has this local contract')
        if not 0 <= fraction <= 1 or tile < 1 or not math.isfinite(max_cache_age) or max_cache_age <= 0:
            raise ValueError('Invalid cache policy')
        if execution not in ('sparse','masked_dense','coalesced'):
            raise ValueError('Unsupported execution')
        self.core, self.pillar_vfe, self.scatter = base.core, base.pillar_vfe, base.scatter
        self.bounds, self.memory_policy, self.variant = base.bounds, base.memory_policy, base.variant
        self.fraction, self.tile, self.max_cache_age = fraction, tile, max_cache_age
        self.execution, self.allow_fallback = execution, allow_fallback
        self.stages = [PackedPyramidStage(getattr(self.core, enc), getattr(self.core, update)).eval()
            for enc, update in [('observation','fine_update'),('coarse_encoder','coarse_update'),('deep_encoder','deep_update')]]
        self.train(False)

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if self.training or torch.is_grad_enabled() or spatial.shape[0] != 1:
            raise ValueError('Frozen single-stream inference required')
        if requests is not None and len(requests)!=3:
            raise ValueError('Three explicit masks required for same-mask diagnostics')
        self.last_requests=[]
        stamp = int(meta['timestamp_us'])
        reason = 'initial' if state is None else None
        if meta.get('reset_before', False):reason = 'explicit'
        if state is not None and reason is None:
            if state['scene'] != meta['segment']:reason = 'scene'
            else:
                gap = (stamp-state['timestamp_us'])/1e6
                if gap <= 0:raise ValueError('Noncausal state')
                if gap > .25:reason = 'gap'
        x = spatial; values = []; references = []; ages = []; stats = []
        for index,(key,stage) in enumerate(zip(self.state_keys,self.stages)):
            shape = ((x.shape[-2]+1)//2,(x.shape[-1]+1)//2)
            evidence = F.avg_pool2d(x,3,2,1,count_include_pad=False)
            if reason:
                old = x.new_zeros(1,stage.output_channels,*shape)
                valid = torch.zeros_like(old[:,:1],dtype=torch.bool)
                reference = evidence; age = torch.zeros_like(old[:,:1])
            else:
                bounds = block_lidar_range(self.bounds,spatial.shape[-2:],shape,2**(index+1))
                aligned,valid = warp_bev(torch.cat((state[key],state['references'][index],state['ages'][index]),1),
                                        state['pose'],meta['pose'],bounds)
                old = aligned[:,:stage.output_channels]
                reference = aligned[:,stage.output_channels:-1]
                age = aligned[:,-1:] + (stamp-state['timestamp_us'])/1e6
            packed = stage.pack(x,old,valid)
            if reason:
                x = stage(packed); written = torch.ones_like(valid)
                requested=written
                stat = dict(actual_write_fraction=1.,executed_input_area_ratio=1.,fallback=False,full_update=True)
            else:
                requested = (change_request(evidence,reference,valid,age,self.fraction,self.tile,self.max_cache_age)
                             if requests is None else requests[index])
                if requested.shape!=valid.shape or requested.dtype!=torch.bool:
                    raise ValueError('Cache mask shape/dtype mismatch')
                if bool(((~valid | (age>=self.max_cache_age-1e-6)) & ~requested).any()):
                    raise ValueError('Explicit mask omits mandatory refresh cells')
                if self.execution == 'coalesced':
                    from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh
                    x,written,stat = coalesced_refresh(stage,packed,old,requested,self.tile)
                else:
                    x,written,stat = refresh_tiles(stage,packed,old,requested,2,stage.radius,self.tile,
                        allow_fallback=self.allow_fallback,execution=self.execution)
                stat.update(full_update=bool(written.all()),mandatory_fraction=float((~valid | (age>=self.max_cache_age-1e-6)).float().mean()))
            values.append(x.detach())
            references.append(torch.where(written,evidence,reference).detach())
            ages.append(torch.where(written,torch.zeros_like(age),age).detach())
            stats.append(dict(stage=index,input_radius=stage.radius,**stat))
            self.last_requests.append(requested.detach().clone())
        ups = [layer(value) for layer,value in zip(self.core.deblocks,values)]
        features = self.core.readout(torch.cat(ups,1))
        prediction = dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:prediction['dm']=self.core.dir_head(features)
        new = {key:value for key,value in zip(self.state_keys,values)}
        new.update(references=references,ages=ages,pose=meta['pose'],scene=meta['segment'],timestamp_us=stamp)
        size = sum(v.numel()*v.element_size() for v in values)
        auxiliary = sum(v.numel()*v.element_size() for v in references+ages)
        return prediction,new,dict(reset=reason,state_bytes=size,cache_bytes=size+auxiliary,
            cache_stages=stats,dense_components=['Pillar','selector','warp','deblocks','readout','heads'])

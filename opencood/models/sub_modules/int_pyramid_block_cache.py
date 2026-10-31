"""Complete per-block caches: a new inference policy, not stage-cache parity."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_pyramid_cache import CachedPyramidDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_local_state_cache import change_request
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh


class LocalBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.stride = block.conv1.stride[0]
        self.radius = 1+self.stride
        self.channels = block.conv2.out_channels

    def forward(self, x):
        return self.block(x)


class LocalUpdate(nn.Module):
    stride = 1
    radius = 2

    def __init__(self, update):
        super().__init__()
        self.update = update
        self.channels = update.history.out_channels

    def forward(self, packed):
        c = self.channels
        return self.update(packed[:,:c],packed[:,c:2*c],packed[:,2*c:] > 0)


class BlockCachedPyramidDetector(CachedPyramidDetector):
    def __init__(self, base, force_full_blocks=(), compact_dense=False, **kwargs):
        super().__init__(base, **kwargs)
        self.blocks = []
        self.output_indices = []
        scale = 1
        for encoder,update in [('observation','fine_update'),('coarse_encoder','coarse_update'),('deep_encoder','deep_update')]:
            for block in getattr(self.core,encoder):
                local = LocalBlock(block).eval(); scale *= local.stride
                self.blocks.append((local,scale,False))
            self.blocks.append((LocalUpdate(getattr(self.core,update)).eval(),scale,True))
            self.output_indices.append(len(self.blocks)-1)
        self.force_full_blocks=frozenset(force_full_blocks)
        if any(type(i) is not int or i<0 or i>=len(self.blocks) for i in self.force_full_blocks):
            raise ValueError('Invalid forced-full block index')
        if compact_dense and self.force_full_blocks!=frozenset(range(9)):
            raise ValueError('Compact layout requires exactly the late-only policy')
        self.compact_dense=compact_dense

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if self.training or torch.is_grad_enabled() or spatial.shape[0]!=1:
            raise ValueError('Frozen single-stream inference required')
        if requests is not None and len(requests)!=len(self.blocks):
            raise ValueError('One diagnostic mask per cached block required')
        stamp=int(meta['timestamp_us']);reason='initial' if state is None else None
        if meta.get('reset_before',False):reason='explicit'
        if state is not None and reason is None:
            if state['scene']!=meta['segment']:reason='scene'
            else:
                gap=(stamp-state['timestamp_us'])/1e6
                if gap<=0:raise ValueError('Noncausal state')
                if gap>.25:reason='gap'
        caches=[];stats=[];values=[];self.last_requests=[];x=spatial
        for i,(module,scale,is_update) in enumerate(self.blocks):
            if self.compact_dense and i<9:
                if is_update:
                    key=self.state_keys[len(values)]
                    if reason:
                        old=torch.zeros_like(x);valid=torch.zeros_like(x[:,:1],dtype=torch.bool)
                    else:
                        bounds=block_lidar_range(self.bounds,spatial.shape[-2:],x.shape[-2:],scale)
                        old,valid=warp_bev(state[key],state['pose'],meta['pose'],bounds)
                    x=module(torch.cat((x,old,valid.to(x.dtype)),1));values.append(x.detach())
                else:x=module(x)
                caches.append(None)
                stats.append(dict(block=i,scale=scale,is_update=is_update,input_radius=module.radius,
                    full_update=True,mandatory_fraction=0.,actual_write_fraction=1.,
                    executed_input_area_ratio=1.,fallback=False,compact_dense=True))
                self.last_requests.append(torch.ones_like(x[:,:1],dtype=torch.bool))
                continue
            stride=module.stride
            evidence=F.avg_pool2d(x,3,2,1,count_include_pad=False) if stride==2 else x
            shape=evidence.shape[-2:]
            if reason:
                old=x.new_zeros(1,module.channels,*shape)
                valid=torch.zeros_like(old[:,:1],dtype=torch.bool)
                reference=evidence;age=torch.zeros_like(old[:,:1])
            else:
                previous=state['blocks'][i]
                bounds=block_lidar_range(self.bounds,spatial.shape[-2:],shape,scale)
                aligned,valid=warp_bev(torch.cat((previous['value'],previous['reference'],previous['age']),1),
                    state['pose'],meta['pose'],bounds)
                old=aligned[:,:module.channels];reference=aligned[:,module.channels:-1]
                age=aligned[:,-1:]+(stamp-state['timestamp_us'])/1e6
            packed=torch.cat((x,old,valid.to(x.dtype)),1) if is_update else x
            mandatory=~valid | (age>=self.max_cache_age-1e-6)
            requested=(torch.ones_like(valid) if reason or i in self.force_full_blocks else
                change_request(evidence,reference,valid,age,self.fraction,self.tile,self.max_cache_age)
                if requests is None else requests[i])
            if requested.shape!=valid.shape or requested.dtype!=torch.bool or bool((mandatory & ~requested).any()):
                raise ValueError('Invalid mask or omitted mandatory cells')
            if reason:
                x=module(packed);written=requested
                info=dict(actual_write_fraction=1.,executed_input_area_ratio=1.,fallback=False)
            elif self.execution in ('coalesced','coalesced_dense'):
                x,written,info=coalesced_refresh(module,packed,old,requested,self.tile,stride=stride,
                    dense_reference=self.execution=='coalesced_dense')
            else:
                x,written,info=refresh_tiles(module,packed,old,requested,stride,module.radius,self.tile,
                    allow_fallback=self.allow_fallback,execution=self.execution)
            caches.append(dict(value=x.detach(),reference=torch.where(written,evidence,reference).detach(),
                age=torch.where(written,torch.zeros_like(age),age).detach()))
            if is_update:values.append(x.detach())
            stats.append(dict(block=i,scale=scale,is_update=is_update,input_radius=module.radius,
                full_update=bool(written.all()),mandatory_fraction=float(mandatory.float().mean()),**info))
            self.last_requests.append(requested.detach().clone())
        features=self.core.readout(torch.cat([layer(value) for layer,value in zip(self.core.deblocks,values)],1))
        prediction=dict(psm=self.core.cls_head(features),rm=self.core.reg_head(features))
        if self.core.dir_head is not None:prediction['dm']=self.core.dir_head(features)
        new={key:value for key,value in zip(self.state_keys,values)}
        new.update(blocks=caches,pose=copy.deepcopy(meta['pose']),scene=meta['segment'],timestamp_us=stamp)
        size=sum(c['value'].numel()*c['value'].element_size() for c in caches if c is not None)
        total=sum(v.numel()*v.element_size() for c in caches if c is not None for v in c.values())
        if self.compact_dense:
            dense_states=sum(v.numel()*v.element_size() for v in values[:2])
            size+=dense_states;total+=dense_states
        return prediction,new,dict(reset=reason,state_bytes=size,cache_bytes=total,cache_stages=stats,
            dense_components=['Pillar','selectors','warps','deblocks','readout','heads'])

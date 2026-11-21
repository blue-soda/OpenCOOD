"""Exact event maintenance for a stride-two depthwise convolution on causal crop jobs."""
from collections import Counter

import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache, PackedLightDeepStage


class CrossScaleDepthwise:
    """Store only globally valid output stencils from the previous ordinary scan."""
    def __init__(self, conv, mode, quantum=.25):
        if (mode not in ('dense','amplitude','unit','static','sparse_amplitude','sparse_unit') or conv.kernel_size!=(3,3)
                or conv.stride!=(2,2) or conv.padding!=(1,1) or conv.dilation!=(1,1)
                or conv.groups!=conv.in_channels or conv.out_channels!=conv.in_channels):
            raise ValueError('Expected a 3x3 stride-two depthwise receiver')
        self.mode=mode
        w=conv.weight.detach().double().reshape(conv.out_channels,9)
        peak=w.abs().amax(dim=1)
        scale=torch.where(peak>0,peak/32767,torch.ones_like(peak))
        self.weight=(w/scale[:,None]).round().short()
        self.scale=(scale*quantum).float()
        self.bias=conv.bias.detach() if conv.bias is not None else torch.zeros_like(self.scale)

    def begin(self, levels, events, input_route, output_route, previous):
        self.levels=levels[0].flatten(1).T
        self.events=events[0].flatten(1).T
        self.h,self.w=levels.shape[-2:]
        self.oh,self.ow=(self.h+1)//2,(self.w+1)//2
        self.c=levels.shape[1]
        idx,missing=input_route
        self.input_map=F.pad(idx.masked_fill(missing.flatten(),-1),(0,1),value=-1)
        idx,missing=output_route
        self.output_map=idx.masked_fill(missing.flatten(),-1)
        self.previous=previous
        size=self.oh*self.ow
        self.cache=(levels.new_zeros(size,self.c),torch.zeros(size,dtype=torch.bool,device=levels.device))
        self.counts=Counter()

    def stencil(self, ids):
        y=ids.div(self.ow,rounding_mode='floor')*2
        x=(ids%self.ow)*2
        axis=torch.arange(-1,2,device=ids.device)
        return y[:,None,None]+axis[None,:,None],x[:,None,None]+axis[None,None,:]

    def lookup(self, data, taps):
        # Out-of-image or crop padding is zero. Shape is sites x taps x channels.
        return data[taps.clamp_min(0)].masked_fill(taps[:,:,None]<0,0)

    def readout(self, linear):
        return linear.float()*self.scale+self.bias

    def region(self, box):
        y0,x0,y1,x1=box
        assert y0%2==x0%2==0
        ys=torch.arange(y0//2,(y1+1)//2,device=self.levels.device)
        xs=torch.arange(x0//2,(x1+1)//2,device=self.levels.device)
        ids=(ys[:,None]*self.ow+xs).flatten()
        y,x=self.stencil(ids)
        inside=(y>=y0)&(y<y1)&(x>=x0)&(x<x1)
        global_inside=(y>=0)&(y<self.h)&(x>=0)&(x<self.w)
        complete=(inside==global_inside).flatten(1).all(1)
        taps=(y*self.w+x).masked_fill(~inside,-1).flatten(1)
        src=self.output_map[ids]
        oy,ox=self.stencil(src.clamp_min(0))
        old_taps=(oy*self.w+ox).masked_fill((oy<0)|(oy>=self.h)|(ox<0)|(ox>=self.w),-1).flatten(1)
        same=complete & (src>=0) & (self.input_map[taps]==old_taps).all(1)
        available=torch.zeros_like(same)
        if self.previous is not None:
            available=(src>=0)&self.previous[1][src.clamp_min(0)]
        reuse=same & available
        ledger=self.counts
        for key,v in [('complete',complete),('mapped_cached',available),('geometry_ok',same),('reuse',reuse)]:
            ledger[key]+=int(v.sum())
        ledger['sites']+=len(ids)
        linear=self.levels.new_empty(len(ids),self.c)
        weight=self.weight.T.long()
        if self.mode=='dense':
            # Independent dense local convolution, with native padding on the original crop.
            crop=self.levels.T.reshape(1,self.c,self.h,self.w)[...,y0:y1,x0:x1].double()
            ref=F.conv2d(crop,self.weight.double().reshape(self.c,1,3,3),stride=2,padding=1,groups=self.c)
            linear=ref[0].flatten(1).T.long().contiguous()
        else:
            cold=torch.ones_like(reuse) if self.mode=='static' else ~reuse
            if bool(cold.any()):
                v=self.lookup(self.levels,taps[cold])
                if self.mode in ('static','sparse_amplitude','sparse_unit'):
                    site,tap,channel=v.nonzero(as_tuple=True)
                    result=v.new_zeros(v.shape[0],self.c)
                    result.view(-1).index_add_(0,site*self.c+channel,v[site,tap,channel]*weight[tap,channel])
                    linear[cold]=result
                    ledger['cold_nonzero_taps']+=len(site)
                else:
                    linear[cold]=(v*weight[None]).sum(1)
                    ledger['cold_nonzero_taps']+=int(v.count_nonzero())
            if self.mode!='static' and bool(reuse.any()):
                changes=self.lookup(self.events,taps[reuse])
                site,tap,channel=changes.nonzero(as_tuple=True)
                amp=changes[site,tap,channel]
                slot=site*self.c+channel
                old=self.previous[0][src[reuse]].contiguous()
                coeff=self.weight[channel,tap].long()
                if self.mode in ('amplitude','sparse_amplitude'):
                    old.view(-1).index_add_(0,slot,amp*coeff)
                else:
                    pos=torch.arange(len(amp),device=amp.device).repeat_interleave(amp.abs())
                    old.view(-1).index_add_(0,slot[pos],(amp.sign()*coeff)[pos])
                linear[reuse]=old
                ledger['delta_nonzero_taps']+=len(amp)
                ledger['positive_units']+=int(amp.clamp_min(0).sum())
                ledger['negative_units']+=int((-amp).clamp_min(0).sum())
        self.cache[0][ids[complete]]=linear[complete]
        self.cache[1][ids[complete]]=True
        pre=self.readout(linear)
        if self.mode=='amplitude':
            # Diagnostics for a possible subsequent 1x1 receiver; never feed these counts back.
            nonzero=int((pre>0).sum())
            changes=nonzero
            if bool(available.any()):
                prior=self.readout(self.previous[0][src[available]]).relu()
                live=pre[available].relu()
                changes+=int((live!=prior).sum())-int(live.count_nonzero())
            ledger['relu_nonzero']+=nonzero
            ledger['relu_changes_or_cold_nonzero']+=changes
        return pre.T.reshape(1,self.c,len(ys),len(xs))

    def end(self):
        result=self.cache
        self.levels=self.events=self.previous=self.cache=None
        return result


class CrossScaleStage(PackedLightDeepStage):
    def __init__(self, encoder, update, mode):
        super().__init__(encoder,update)
        self.edge=CrossScaleDepthwise(encoder[0],mode)

    def region_inputs(self,current,old,valid,jobs):
        boxes=[(0,0,*current.shape[-2:])] if jobs is None else [j[4:8] for j in jobs]
        def crop(t):
            views=[t[...,y//2:(ey+1)//2,x//2:(ex+1)//2] for y,x,ey,ex in boxes]
            return views[0] if len(views)==1 else torch.cat(views)
        values=[self.edge.region(box) for box in boxes]
        return (values[0] if len(values)==1 else torch.cat(values)),crop(old),crop(valid)

    def forward(self, preactivation, old, valid):
        return self.update(self.encoder[2](self.encoder[1](preactivation)),old,valid)


class CrossScaleCache(DeepBlockCache):
    """Middle signed messages feed the ordinary deep observation before ReLU."""
    def __init__(self, source, mode, **kwargs):
        super().__init__(source,8,'light',crop_first=True,**kwargs)
        self.stage=CrossScaleStage(self.core.deep_light,self.core.deep_update,mode).eval()

    def prepare(self,spatial,state,meta):
        frame=super().prepare(spatial,state,meta)
        sender=self.coarse_projector
        if sender.predictor!='zero' or sender.den!=1:
            raise ValueError('This receiver requires aligned, amplitude-preserving middle levels')
        self.stage.edge.begin(sender.numerator,sender.events,frame.coarse_route,frame.deep_route,state.get('edge_cache'))
        return frame

    def step_spatial(self,spatial,state,meta,requests=None,dense_reference=False):
        pred,new,info=super().step_spatial(spatial,state,meta,requests,dense_reference)
        if info['full_refresh'] is None:
            edge=self.stage.edge
            new['edge_cache']=edge.end()
            info['edge']=dict(edge.counts)
        return pred,new,info

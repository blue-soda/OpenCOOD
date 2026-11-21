"""Reserve part of the shallow wire budget for task-conditioned feature events."""
import numpy as np
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_grouped_receiver import GroupedEdge,GroupedReceiver,take_grouped
from opencood.models.sub_modules.int_silent_receiver import Candidates,SilentReceiver


class ReservedReceiver(GroupedReceiver):
    def __init__(self,lower=False,shuffle=False,floating=False):
        super().__init__(6 if lower else 12,floating=floating)
        self.lower,self.shuffle=lower,shuffle

    def guide(self,risk,frame):
        self.risk=risk.repeat_interleave(2,-2).repeat_interleave(2,-1)
        self.frame=int(frame)

    def prepare(self,value,sites,warp):
        c=super().prepare(value,sites,warp)
        h,w=value.shape[-2:]
        assert self.risk.shape[-2:]==(h,w)
        width=(w+self.side-1)//self.side
        def group(pos):
            return (pos//w//self.side)*width+(pos%w)//self.side
        read=torch.zeros_like(self.risk,dtype=torch.bool)
        read.view(-1)[sites]=True
        pool=lambda x:F.max_pool2d(x,self.side,self.side,ceil_mode=True).flatten()
        score=pool(self.risk.masked_fill(~read,0)).cpu().numpy()
        valid=pool(read.float()).bool().cpu().numpy()
        tiles=np.flatnonzero(valid)
        if self.shuffle:
            score[tiles]=np.random.RandomState(self.frame+303).permutation(score[tiles])
        order=np.lexsort((np.arange(len(score)),-score))
        self.order=torch.as_tensor(order,device=sites.device)
        high=torch.zeros(len(score),dtype=torch.bool,device=sites.device)
        high[torch.as_tensor(order[valid[order]][:(len(tiles)+3)//4],device=sites.device)]=True
        self.groups=group(c.ids%(h*w))
        standard=c.change.abs()>=(3 if self.floating else 12)
        keep=standard | high[self.groups]
        self.groups=self.groups[keep]
        self.standard=standard[keep]
        return Candidates(c.levels,c.ids[keep],c.change[keep])

    def receive(self,c,limit,frame,read,book):
        main=self.standard.nonzero().flatten()
        tile=self.groups
        mass=torch.zeros(len(self.order),device=c.ids.device,dtype=torch.long)
        mass.scatter_add_(0,tile[main],self.magnitude(c.change[main]))
        tile_ids=torch.arange(len(mass),device=c.ids.device)
        order=torch.argsort(mass*len(mass)+len(mass)-1-tile_ids,descending=True)
        first=main[take_grouped(c.ids[main],c.change[main],tile[main],order,limit*7//8,frame)]
        remaining=torch.ones(len(c.ids),device=c.ids.device,dtype=torch.bool)
        remaining[first]=False
        rest=remaining.nonzero().flatten()
        second=rest[take_grouped(c.ids[rest],c.change[rest],tile[rest],self.order,limit-len(first),frame)]
        chosen=torch.cat((first,second))
        assert len(chosen)==limit
        value,stats=SilentReceiver.receive(self,Candidates(c.levels,c.ids[chosen],c.change[chosen]),limit,frame,read,book)
        stats.update(eligible=len(c.ids),ordinary_candidates=len(main),reserved=len(second),
                     low_sent=int((~self.standard[chosen]).sum()))
        return value,stats


class ReservedEdge(GroupedEdge):
    def __init__(self,bounds,cap,book):
        super().__init__(bounds,cap,book)
        self.arms=dict(tile=GroupedReceiver(12),guided=ReservedReceiver(),lower=ReservedReceiver(lower=True),
                       shuffle=ReservedReceiver(lower=True,shuffle=True),float_bytes=ReservedReceiver(lower=True,floating=True))
        for arm in self.arms.values():arm.capture_sites=True

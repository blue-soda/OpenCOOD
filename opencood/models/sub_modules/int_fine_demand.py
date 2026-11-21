"""Deliver fine-level changes only to the receptive field of real middle jobs."""
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import supports
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_edge import FineEdgeSender


def fine_support(middle_active, fine_shape):
    # The middle residual block requires two rings. Its first observation uses
    # a stride-two, 3x3 padded convolution centered on even fine coordinates.
    middle = supports(middle_active)[2]
    centers = middle.new_zeros(1,1,*fine_shape)
    centers[...,::2,::2] = middle
    return F.max_pool2d(centers.float(),3,1,1).bool()


class DemandFineEdge(FineEdgeSender):
    def __init__(self,bounds,defer):
        super().__init__(bounds,send_on='ordinary')
        self.defer=defer

    def input_to_middle(self,value):
        # At refresh we initialize the sender and pass the original continuous
        # value to the full encoder. On ordinary scans select jobs from raw fine.
        if self.phase=='refresh':
            return self(value)
        return value

    def receive_jobs(self,value,active):
        if not self.defer:
            return self(value)
        selected=fine_support(active,value.shape[-2:])
        sites=selected.flatten().nonzero().flatten()
        area=block_lidar_range(self.bounds,self.shape,value.shape[-2:],2)
        def align(x):
            return warp_bev(x,self.previous['pose'],self.pose,area)[0]
        aligned=align(self.previous['fine_levels'].float()).int().contiguous()
        channels=value.shape[1]
        receiver=aligned.view(channels,-1)
        local=(value.reshape(channels,-1)[:,sites]/self.quantum).round()
        if not bool(local.abs().max()<2**23):
            raise ValueError('Fine level out of exact range')
        target=local.int()
        delta=target-receiver[:,sites]
        channel,slot=delta.nonzero(as_tuple=True)
        change=delta[channel,slot]
        receiver.view(-1).index_add_(0,channel*receiver.shape[1]+sites[slot],change)
        assert torch.equal(receiver[:,sites],target)

        # Diagnostics only: attribute eager traffic to used/unused positions
        # and separate accumulated catch-up from ordinary current innovations.
        previous_eager=align((self.previous['value']/self.quantum).round()).int().view(channels,-1)
        eager_delta=target-previous_eager[:,sites]
        stale=(delta!=eager_delta)
        self.ledger=dict(cells=value.numel(),cache_bytes=aligned.numel()*aligned.element_size(),
                         fine_sites=selected.numel(),read_sites=len(sites),
                         addresses=len(change),units=int(change.abs().sum(dtype=torch.int64)),
                         eager_read_addresses=int(eager_delta.count_nonzero()),eager_read_units=int(eager_delta.abs().sum(dtype=torch.int64)),
                         stale_read_channels=int(stale.sum()),catchup_only=int(((eager_delta==0)&(delta!=0)).sum()),
                         cancelled=int(((eager_delta!=0)&(delta==0)).sum()))
        self.levels=aligned
        self.previous=None
        return aligned.float()*self.quantum

    def finish(self,state):
        if not self.defer or self.phase=='refresh':
            self.commit(state,False)
        else:
            state['fine_levels']=self.levels
            self.levels=None


class QuotaFineEdge(FineEdgeSender):
    """One unit or one signed amplitude per admitted address in the read region."""
    def __init__(self,bounds,mode,cap):
        super().__init__(bounds,send_on='ordinary')
        if mode not in ('unit_carry','unit_round','amplitude') or cap<=0:
            raise ValueError('Unknown admission policy or invalid quota')
        self.mode=mode
        self.cap=cap

    def begin(self,previous,meta,spatial_shape,reason):
        super().begin(previous,meta,spatial_shape,reason)
        self.frame_number=int(meta['frame'])
        self.active=None

    def __call__(self,value):
        if self.phase=='refresh':
            return super().__call__(value)
        self.read=fine_support(self.active,value.shape[-2:])
        sites=self.read.flatten().nonzero().flatten()
        area=block_lidar_range(self.bounds,self.shape,value.shape[-2:],2)
        aligned=warp_bev(self.previous['fine_levels'].float(),self.previous['pose'],self.pose,area)[0].int().contiguous()
        receiver=aligned.view(value.shape[1],-1)
        target=(value.flatten(2)[0,:,sites]/self.quantum).round().int()
        delta=target-receiver[:,sites]
        channel,slot=delta.nonzero(as_tuple=True)
        ids=channel*receiver.shape[1]+sites[slot]
        change=delta[channel,slot]
        demand=len(ids)
        if demand>self.cap:
            tie=((ids+self.frame_number)*2654435761)%(1<<32)
            key=change.long().abs()*(1<<32)+tie
            chosen=torch.topk(key,self.cap,sorted=False).indices
            ids,change=ids[chosen],change[chosen]
        change,extra=self.deliver(receiver,ids,change)
        self.levels=aligned
        self.previous=None
        self.ledger=dict(addresses=len(ids),units=int(change.abs().sum(dtype=torch.int64)),
                         demand_addresses=demand, cells=value.numel(),cache_bytes=aligned.numel()*aligned.element_size())
        self.ledger.update(extra)
        return aligned.float()*self.quantum

    def deliver(self,receiver,ids,change):
        if self.mode!='amplitude':
            change=change.sign()
        receiver.view(-1).index_add_(0,ids,change)
        return change,{}

    def commit(self,state,drop):
        if self.phase=='refresh':
            return super().commit(state,False)
        state['fine_levels']=self.levels
        if self.mode=='unit_round':
            raw=state['value']
            rounded=(raw/self.quantum).round()*self.quantum
            state['value']=torch.where(self.read,rounded,raw)
        self.read=self.levels=None


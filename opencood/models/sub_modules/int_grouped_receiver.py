"""Select threshold crossings in spatial groups before applying the original wire codec."""
import torch

from opencood.models.sub_modules.int_fine_demand import fine_support
from opencood.models.sub_modules.int_silent_receiver import Candidates, PairedEdge, SilentReceiver


class GroupedReceiver(SilentReceiver):
    def __init__(self, threshold, side=12, **kwargs):
        super().__init__(threshold, **kwargs)
        self.side = side

    def magnitude(self, change):
        # Physical feature units of 1/256, including the four-level sender's 1/4 quantum.
        return ((change.abs()*256).round().long() if self.floating else change.abs().long()*64)

    def tile_mass(self, count, tile, change, read):
        mass = torch.zeros(count,device=tile.device,dtype=torch.long)
        return mass.scatter_add_(0,tile,self.magnitude(change))

    def receive(self, candidates, limit, frame, read, book):
        ids, change = candidates.ids, candidates.change
        demand = ids.numel()
        if 0 < limit < demand:
            height, width = read.shape[-2:]
            sites = ids % (height*width)
            columns = (width+self.side-1)//self.side
            tile = (sites//width//self.side)*columns + sites%width//self.side
            count = columns*((height+self.side-1)//self.side)
            mass = self.tile_mass(count,tile,change,read)
            tile_ids = torch.arange(count,device=ids.device)
            order = torch.argsort(mass*count + count-1-tile_ids,descending=True)
            sizes = torch.bincount(tile,minlength=count)
            end = sizes[order].cumsum(0)
            cut = int(torch.searchsorted(end,limit))
            full = torch.zeros(count,device=ids.device,dtype=torch.bool)
            full[order[:cut]] = True
            keep = full[tile].nonzero().flatten()
            last = (tile==order[cut]).nonzero().flatten()
            take = limit-len(keep)
            if take < len(last):
                magnitude = change[last].abs().contiguous()
                if self.floating:
                    magnitude = magnitude.view(torch.int32)
                tie = ((ids[last]+frame)*2654435761) % (1<<32)
                last = last[torch.topk(magnitude.long()*(1<<32)+tie,take,sorted=False).indices]
            selected = torch.cat((keep,last))
            assert len(selected)==limit
            candidates = Candidates(candidates.levels,ids[selected],change[selected])
        value, stats = super().receive(candidates,limit,frame,read,book)
        stats['eligible'] = demand
        return value,stats


class ChargedReceiver(GroupedReceiver):
    """Carry unserved spatial priority until that location actually transmits."""
    def refresh(self, value):
        super().refresh(value)
        self.debt = torch.zeros_like(value[:,:1],dtype=torch.long)

    def prepare(self, value, sites, warp):
        # The existing BEV warp temporarily casts to FP32.
        if int(self.debt.max()) >= 2**24:
            raise ValueError('Unserved spatial charge is outside the exact warp range')
        self.aligned_debt = warp(self.debt)
        return super().prepare(value,sites,warp)

    def tile_mass(self, count, tile, change, read):
        height,width = read.shape[-2:]
        ids = torch.arange(height*width,device=read.device)
        groups = (ids//width//self.side)*((width+self.side-1)//self.side)+ids%width//self.side
        scores = torch.zeros(count,device=read.device,dtype=torch.long)
        return scores.scatter_add_(0,groups,self.priority.flatten())

    def receive(self, candidates, limit, frame, read, book):
        mass = torch.zeros(read.numel(),device=read.device,dtype=torch.long)
        mass.scatter_add_(0,candidates.ids%read.numel(),self.magnitude(candidates.change))
        active = mass.reshape_as(read)>0
        prior = torch.where(active,self.aligned_debt,0)
        self.priority = prior + mass.reshape_as(read)
        self.capture_sites = True
        value,stats = super().receive(candidates,limit,frame,read,book)
        self.debt = self.priority.masked_fill(self.event_sites,0)
        stats.update(prior_mass=int(prior.sum()),outstanding_mass=int(self.debt.sum()),
                     state_bytes=self.debt.numel()*self.debt.element_size())
        self.priority = self.aligned_debt = None
        return value,stats


class GroupedEdge(PairedEdge):
    def __init__(self, bounds, cap, book, charge=False):
        super().__init__(bounds,cap,book,[12],calibration=False)
        self.arms = dict(point=SilentReceiver(12),tile=GroupedReceiver(12),
                         reset=GroupedReceiver(12,forget=True),float_bytes=GroupedReceiver(12,floating=True))
        if charge:
            self.arms = dict(tile=GroupedReceiver(12),charge=ChargedReceiver(12),
                             float_bytes=GroupedReceiver(12,floating=True),
                             float_bytes_charge=ChargedReceiver(12,floating=True))
        for arm in self.arms.values():
            arm.capture_sites = True

    def apply(self, name, value, active=None):
        first = not self.result
        result = super().apply(name,value,active)
        if first and not self.reason:
            sites = fine_support(active,value.shape[-2:]).flatten().nonzero().flatten()
            target = value.flatten(2)[:,:,sites]
            for key,received in self.result.items():
                stats = self.stats[key]
                diff = (received.flatten(2)[:,:,sites] - target).abs()
                stats['error_sum'] = float(diff.sum(dtype=torch.float64))
                stats['values'] = diff.numel()
        return result

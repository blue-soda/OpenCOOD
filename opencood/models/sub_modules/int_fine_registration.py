"""Use current scan evidence to propagate integer receiver levels before sending corrections."""
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_feature_memory import warp_bev


OFFSETS = [(0,0)] + [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]


def descriptor(value):
    return value.abs().reshape(1,8,-1,*value.shape[-2:]).mean(2)


def estimate(older, last, valid):
    # Common support for all shifts, including border tiles, prevents padding wins.
    h,w = valid.shape[-2:]
    border = F.pad((~valid).float(), (1,1,1,1), value=1)
    valid = F.max_pool2d(border,3,1)==0
    pad = (0,-w%8,0,-h%8)
    def tile_sum(x):
        return F.avg_pool2d(F.pad(x,pad),8,8,divisor_override=1)
    n = tile_sum(valid.float())
    costs = torch.cat([tile_sum((last-torch.roll(older,shift,(-2,-1))).abs().mean(1,keepdim=True)*valid)
                       for shift in OFFSETS],1)
    best, choice = costs.min(1,keepdim=True)
    active = (n>=16) & (choice!=0) & (best<.85*costs[:,:1])
    choice = torch.where(active,choice,torch.zeros_like(choice))
    selected = choice.repeat_interleave(8,-2).repeat_interleave(8,-1)[:,:,:h,:w]
    selected = selected.masked_fill(~valid,0)
    stats = dict(eligible_tiles=int((n>=16).sum()), moved_tiles=int(active.sum()),
                 history_identity=float(costs[:,:1][active].sum()), history_motion=float(best[active].sum()))
    return selected, valid, stats


def tile_packet(selected, valid):
    h, w = selected.shape[-2:]
    padded = F.pad(selected.float(), (0,-w%8,0,-h%8))
    ranks = F.max_pool2d(padded,8,8).byte()
    flat = ranks.flatten()
    pairs = F.pad(flat,(0,len(flat)%2)).view(-1,2)
    packet = pairs[:,0] | (pairs[:,1] << 4)
    unpacked = torch.stack((packet & 15,packet >> 4),-1).flatten()[:len(flat)].view_as(ranks)
    decoded = unpacked.repeat_interleave(8,-2).repeat_interleave(8,-1)[:,:,:h,:w].masked_fill(~valid,0)
    assert torch.equal(selected, decoded.long())
    return packet, decoded


def displace(levels, selected):
    h,w = levels.shape[-2:]
    offsets = selected.new_tensor([dy*w+dx for dy,dx in OFFSETS],dtype=torch.long)
    indices = torch.arange(h*w,device=levels.device) - offsets[selected.long().flatten()]
    return levels.flatten(2).index_select(2,indices).view_as(levels)


class ReceiverGuide:
    def __init__(self):
        self.current = None
        self.cost = {}

    def observe(self, module, inputs, output):
        if self.current is not None:
            raise RuntimeError('Expected exactly one fresh observation per detector step')
        self.current = descriptor(output.detach())

    def propagate(self, levels, previous, pose, area):
        old,valid = warp_bev(previous['observation_descriptor'],previous['pose'],pose,area)
        selected,valid,stats = estimate(old,self.current,valid)
        packet,decoded = tile_packet(selected,valid)
        self.cost = dict(tiles=stats['eligible_tiles'],moved_tiles=stats['moved_tiles'],
                         moved_sites=int(decoded.count_nonzero()), packet_bytes=packet.numel(),
                         descriptor_bytes=old.numel()*old.element_size(),
                         match_components=9*old.numel())
        return displace(levels,decoded)

    def save(self, state):
        if self.current is None:
            raise RuntimeError('Missing fresh scan descriptor')
        state['observation_descriptor'] = self.current
        self.current = None


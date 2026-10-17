"""Dependency-safe request masks for local repair; execution remains separate."""
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles


def affected_outputs(changed_input, stride, radius):
    """Outputs whose receptive field touches a changed input position."""
    return F.max_pool2d(changed_input.float(), 2*radius+1, stride, radius) > 0


def local_change_mask(current, aligned, valid, threshold):
    pool = lambda x: F.avg_pool2d(x,4,4,ceil_mode=True,count_include_pad=False)
    error = pool((current-aligned).square().mean(1,keepdim=True))
    energy = torch.maximum(pool(current.square().mean(1,keepdim=True)),
                           pool(aligned.square().mean(1,keepdim=True)))
    scores = (error/energy.clamp_min(1e-6)).sqrt()
    return expand_tiles(scores > threshold,4,current.shape[-2:]) | ~valid


def local_requests(changed_stage1, valid_stage2, valid_stage3, tile):
    # Four stage2 BasicBlocks: radius15 in stage1 input. Five stage3: radius19.
    request2 = affected_outputs(changed_stage1,2,15) | ~valid_stage2
    written2 = expand_tiles(tile_mask(request2,tile),tile,request2.shape[-2:])
    # Refresh all stage3 outputs potentially affected by stage2 repair, including
    # boundary repairs. Just resizing the original change mask misses this halo.
    request3 = affected_outputs(written2,2,19) | ~valid_stage3
    written3 = expand_tiles(tile_mask(request3,tile),tile,request3.shape[-2:])
    return written2,written3

"""Coalesce execution regions while preserving the original tile write mask."""
from contextlib import nullcontext

import torch
from opencood.models.sub_modules.int_cache_repair import (
    tile_mask, expand_tiles, rectangles, crop_jobs)


def plan_coalesced_refresh(requested, input_shape, radius, tile, stride=2):
    """Plan the exact write mask and count executed sites, including halos."""
    output_shape = requested.shape[-2:]
    written = expand_tiles(tile_mask(requested, tile), tile, output_shape)
    layouts = ((size, rectangles(tile_mask(written, size)))
               for size in (tile, tile*2, tile*4, tile*8, tile*16))
    return (written, *_choose_jobs(layouts, input_shape, output_shape, radius, stride))


def plan_host_coalesced_refresh(requested, input_shape, radius, tile, stride=2):
    """Transfer the base tile grid once and evaluate coarsenings on the host."""
    output_shape = requested.shape[-2:]
    active = tile_mask(requested, tile)
    written = expand_tiles(active, tile, output_shape)
    host_tiles = active.cpu()
    layouts = ((tile*factor, rectangles(host_tiles if factor == 1 else tile_mask(host_tiles, factor)))
               for factor in (1, 2, 4, 8, 16))
    return (written, *_choose_jobs(layouts, input_shape, output_shape, radius, stride))


def _choose_jobs(layouts, input_shape, output_shape, radius, stride):
    h, w = input_shape
    oh, ow = output_shape
    dense_work = oh*ow
    best_jobs = None; best_work = dense_work; chosen_tile = None
    original_work = None
    for size, regions in layouts:
        jobs, _ = crop_jobs(regions, h, w, oh, ow, stride, radius, size)
        work = sum(((j[6]-j[4]+stride-1)//stride)*((j[7]-j[5]+stride-1)//stride) for j in jobs)
        if original_work is None: original_work = work
        if work < best_work:
            best_work, best_jobs, chosen_tile = work, jobs, size
    return best_jobs, best_work, original_work, chosen_tile


def coalesced_refresh(module, x, cached, requested, tile=4, batch_size=8, stride=2, dense_reference=False,
                      planner=plan_coalesced_refresh, phase=nullcontext, region_inputs=None):
    """Local blocks whose convolutions all use the same output grid.

    All convolutions have the same output lattice, so executed sites are
    proportional to actual MACs. Dense fallback preserves the write mask.
    """
    if module.training or x.shape[0] != 1 or batch_size < 1 or tile < 1 or stride not in (1,2):
        raise ValueError('Frozen single-stream inference required')
    h, w = x.shape[-2:]; oh, ow = cached.shape[-2:]
    if (oh, ow) != ((h+stride-1)//stride, (w+stride-1)//stride) or requested.shape != cached[:, :1].shape:
        raise ValueError('Output/mask mismatch')
    with phase('region_plan'):
        written, best_jobs, best_work, original_work, chosen_tile = planner(
            requested, (h, w), module.radius, tile, stride)
    dense_work = oh * ow
    with phase('stats_sync'):
        stats = dict(requested_fraction=float(requested.float().mean()),
                     selected_fraction=float(written.float().mean()),
                     actual_write_fraction=float(written.float().mean()),
                     fallback=best_jobs is None, execution_tile=chosen_tile,
                     original_stage_conv_ratio=original_work/float(dense_work),
                     stage_conv_ratio=best_work/float(dense_work),
                     tiles=1 if best_jobs is None else len(best_jobs))
    if best_jobs is None or dense_reference:
        stats['executed_input_area_ratio'] = 1.
        stats['dense_reference'] = dense_reference
        if dense_reference:
            stats['planned_stage_conv_ratio'] = stats['stage_conv_ratio']
            stats['stage_conv_ratio'] = 1.
        with phase('deep_compute'):
            result = module(x) if region_inputs is None else module(*region_inputs(None))
        with phase('write_back'):
            return torch.where(written, result, cached), written, stats
    stats['executed_input_area_ratio'] = sum((j[6]-j[4])*(j[7]-j[5]) for j in best_jobs)/float(h*w)
    with phase('write_back'):
        output = cached.clone()
    with phase('crop_pack'):
        groups = {}
        for job in best_jobs:
            groups.setdefault((job[6]-job[4], job[7]-job[5]), []).append(job)
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batch = group[start:start+batch_size]
            with phase('crop_pack'):
                if region_inputs is None:
                    inputs = (torch.cat([x[..., j[4]:j[6], j[5]:j[7]] for j in batch], 0),)
                else:
                    inputs = region_inputs(batch)
            with phase('deep_compute'):
                result = module(*inputs)
            with phase('write_back'):
                for i, (y0,x0,y1,x1,iy0,ix0,_,_) in enumerate(batch):
                    sy,sx = y0-iy0//stride,x0-ix0//stride
                    mask = written[..., y0:y1, x0:x1]
                    output[..., y0:y1, x0:x1] = torch.where(mask,
                        result[i:i+1,:,sy:sy+y1-y0,sx:sx+x1-x0],cached[...,y0:y1,x0:x1])
    return output, written, stats

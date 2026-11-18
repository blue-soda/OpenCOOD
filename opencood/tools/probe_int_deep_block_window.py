"""Probe the future detection value of a single equal-cost deep-cache intervention."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
from typing import NamedTuple

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, TILE, checked_plan, clip_streams, tile_features
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket, masks_by_tile, spatial_input
from opencood.tools.run_int_ego import to_device, write_json


WINDOW = 3
PROBES_PER_CLIP = 4
POLICIES = ('residual', 'current_oracle', 'window_oracle')
METRICS = ('current_loss', 'future_loss', 'window_loss')


class TraceFrame(NamedTuple):
    meta: dict
    spatial: torch.Tensor
    target: object
    prior: object
    refresh: object
    request: object
    write_mask: object


def fixed_history(cache, ids, stream, trace_log, clip):
    state, trace = None, []
    for index, sample in enumerate(stream):
        meta = sample['meta']
        assert meta['frame'] == ids[index]
        ego = to_device(sample['ego'], torch.device('cuda'))
        spatial = spatial_input(cache.source, ego)
        prior = state
        refresh = cache.refresh_reason(prior, meta)
        request, write_mask, optional, mandatory = None, None, None, None
        if refresh is None:
            frame = cache.prepare(spatial, prior, meta)
            required = cache.mandatory(frame).cpu()
            optional_tiles = list(masks_by_tile(required, cache.tile))
            if optional_tiles:
                optional = optional_tiles[index * 7 % len(optional_tiles)]
                request = add_tiles(torch.zeros_like(required), (optional,), cache.tile).cuda()
            write_mask = required.cuda() if request is None else (required.cuda() | request)
            mandatory = required.flatten().nonzero().flatten().tolist()
        _, state, info = cache.step_spatial(spatial, prior, meta, request)
        assert info['full_refresh'] == refresh
        target = ego['label_dict'] if meta['supervised'] else None
        trace.append(TraceFrame(meta, spatial, target, prior, refresh, request, write_mask))
        trace_log.write(json.dumps(dict(clip=clip, frame=ids[index], refresh=refresh,
            supervised=target is not None, optional_tile=optional, mandatory_flat_indices=mandatory)) + '\n')
    assert len(trace) == len(ids)
    return trace


def evaluate_group(cache, window, pair, required, sites, residual, criterion):
    root = window[0]
    state = root.prior
    request = add_tiles(torch.zeros_like(required), pair, cache.tile).cuda()
    losses, costs, fixed_writes = [], [], []
    for offset, scan in enumerate(window):
        if offset:
            def fixed_request(frame, mandatory):
                combined = mandatory if scan.request is None else (mandatory | scan.request)
                fixed_writes.append(torch.equal(combined, scan.write_mask))
                return scan.request

            request = fixed_request
        (pred, state, info), flops = measured_step(cache, scan.spatial, state, scan.meta, request)
        assert info['full_refresh'] is None
        if not offset:
            assert abs(info['stage_conv_ratio'] - sites / required.numel()) < 1e-12
        loss = None if scan.target is None else float(criterion(pred, scan.target))
        if loss is not None and not math.isfinite(loss):
            raise ValueError('Non-finite detection loss')
        costs.append(flops)
        losses.append(loss)
    return dict(pair=pair, residual_score=sum(float(residual[key]) for key in pair),
                loss_by_scan=losses, core_flops_by_scan=costs, future_write_mask_matches=fixed_writes,
                current_loss=losses[0], future_loss=mean(v for v in losses[1:] if v is not None),
                window_loss=mean(v for v in losses if v is not None))


def probe_windows(cache, trace, criterion, clip):
    candidates = []
    for index in range(len(trace) - WINDOW + 1):
        window = trace[index:index + WINDOW]
        root = window[0]
        if any(scan.refresh is not None for scan in window):
            continue
        if root.target is None or all(scan.target is None for scan in window[1:]):
            continue
        frame = cache.prepare(root.spatial, root.prior, root.meta)
        required = cache.mandatory(frame).cpu()
        base_sites, sites, pairs = group_bucket(cache, frame, required, CAP)
        if len(pairs) < 2:
            continue
        residual = tile_features(frame)[..., 0]
        groups = [evaluate_group(cache, window, pair, required, sites, residual, criterion) for pair in pairs]
        cost_match = len({tuple(g['core_flops_by_scan']) for g in groups}) == 1
        writes_match = all(len(g['future_write_mask_matches']) == WINDOW - 1
                           and all(g['future_write_mask_matches']) for g in groups)
        matched = cost_match and writes_match
        row = dict(clip=clip, frames=[scan.meta['frame'] for scan in window],
                   base_sites=base_sites, sites=sites, lattice_sites=required.numel(),
                   cost_match=cost_match, future_writes_match=writes_match, matched=matched, groups=groups)
        if matched:
            order = range(len(groups))
            row['selected'] = dict(
                residual=max(order, key=lambda i: groups[i]['residual_score']),
                current_oracle=min(order, key=lambda i: groups[i]['current_loss']),
                window_oracle=min(order, key=lambda i: groups[i]['window_loss']),
            )
        candidates.append(row)
        if len(candidates) >= PROBES_PER_CLIP:
            break
    return candidates


def summarize(windows):
    matched = [r for r in windows if r['matched']]
    result = dict(windows=len(windows), matched_windows=len(matched),
                  cost_mismatches=sum(not r['cost_match'] for r in windows),
                  future_write_mismatches=sum(not r['future_writes_match'] for r in windows))
    if not matched:
        return result

    def metric(row, policy, field):
        return row['groups'][row['selected'][policy]][field]

    result['current_window_oracle_choice_changes'] = sum(
        r['selected']['current_oracle'] != r['selected']['window_oracle'] for r in matched)
    result['mean_loss'] = {p:{field:mean(metric(r, p, field) for r in matched) for field in METRICS}
                           for p in POLICIES}
    result['mean_core_flops_by_scan'] = [mean(r['groups'][0]['core_flops_by_scan'][i] for r in matched)
                                        for i in range(WINDOW)]
    result['gains'] = {}
    for left, right in (('residual', 'current_oracle'), ('residual', 'window_oracle'),
                        ('current_oracle', 'window_oracle')):
        result['gains'][right + '_over_' + left] = {}
        for field in METRICS:
            gains = [metric(r, left, field) - metric(r, right, field) for r in matched]
            result['gains'][right + '_over_' + left][field] = dict(mean=mean(gains),
                improves=sum(v > 1e-6 for v in gains), ties=sum(abs(v) <= 1e-6 for v in gains),
                worsens=sum(v < -1e-6 for v in gains))
    return result


def run(args, out):
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train'
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == plan['split']
    source.period, source.max_age = 5, .55
    cache = DeepBlockCache(source, TILE, 'light')
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    original = model_digest(source)
    write_json(out / 'contract.json', dict(plan_sha256=digest(Path(args.plan)), source_sha256=SOURCE_SHA,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), torch=torch.__version__,
        device=torch.cuda.get_device_name(), period=source.period, max_age=source.max_age, tile=TILE,
        cap=CAP, window=WINDOW, max_windows_per_clip=PROBES_PER_CLIP,
        policy='Geometry-only neutral history and fixed future masks; one equal-work candidate intervention at t',
        objective='Exact detector loss on labeled scans; future labels used only by offline window oracle',
        eligibility='Earliest windows with t labeled, >=1 labeled later scan, all ordinary, >=2 equal-work options'))
    all_windows, per_clip = [], []
    with torch.no_grad(), (out / 'frames.jsonl').open('x', buffering=1) as raw, \
            (out / 'trace.jsonl').open('x', buffering=1) as trace_log:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            trace = fixed_history(cache, ids, stream, trace_log, clip)
            windows = probe_windows(cache, trace, criterion, clip)
            for window in windows:
                raw.write(json.dumps(window, allow_nan=False) + '\n')
            per_clip.append(dict(clip=clip, scans=len(trace), **summarize(windows)))
            all_windows.extend(windows)
            del trace
            write_json(out / 'status.json', dict(status='RUNNING', clips=clip + 1, windows=len(all_windows)))
    assert model_digest(source) == original
    summary = summarize(all_windows)
    write_json(out / 'report.json', dict(status='COMPLETED', weight_unchanged=True, per_clip=per_clip,
        scans=sum(map(len, plan['clips'])), parameters=sum(p.numel() for p in source.parameters()),
        ap50=None, ap70=None, model_step_ms_mean=None, **summary))
    write_json(out / 'status.json', dict(status='COMPLETED', windows=len(all_windows),
        matched_windows=summary['matched_windows']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'trained', 'readout', 'plan', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(status='RUNNING', phase='inputs'))
    try:
        run(args, out)
    except Exception as exc:
        write_json(out / 'status.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

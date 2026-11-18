"""Measure the effect of an equal-cost deep write across one scheduled refresh."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, TILE, checked_plan, clip_streams, tile_features
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket
from opencood.tools.probe_int_deep_block_window import PROBES_PER_CLIP, WINDOW, fixed_history, summarize
from opencood.tools.run_int_ego import write_json


EPSILON = 1e-6
PHASES = ('pre_refresh_loss', 'post_refresh_loss')


def periodic_offset(window):
    if len(window) != WINDOW or window[0].refresh is not None or window[0].target is None:
        return None
    refresh = [i for i, scan in enumerate(window) if scan.refresh is not None]
    if len(refresh) != 1 or window[refresh[0]].refresh != 'period':
        return None
    if not any(scan.target is not None for scan in window[refresh[0]:]):
        return None
    return refresh[0]


def evaluate_group(cache, window, pair, required, sites, residual_score, criterion, refresh_at, reference=None):
    state = window[0].prior
    losses, costs, mask_matches, deep = [], [], [], []
    for offset, scan in enumerate(window):
        request = None
        if offset == 0:
            request = add_tiles(torch.zeros_like(required), pair, cache.tile).cuda()
        elif scan.refresh is None:
            def same_future_write(frame, mandatory):
                write_mask = mandatory if scan.request is None else (mandatory | scan.request)
                mask_matches.append(torch.equal(write_mask, scan.write_mask))
                return scan.request

            request = same_future_write
        (pred, state, info), flops = measured_step(cache, scan.spatial, state, scan.meta, request)
        assert info['full_refresh'] == scan.refresh, scan.meta['frame']
        if offset == 0:
            assert abs(info['stage_conv_ratio'] - sites / required.numel()) < 1e-12
        loss = None if scan.target is None else float(criterion(pred, scan.target))
        if loss is not None and not math.isfinite(loss):
            raise ValueError('Non-finite detection loss at ' + scan.meta['frame'])
        costs.append(flops)
        losses.append(loss)
        deep.append(state['deep'])
    delta = [0.] * WINDOW if reference is None else [float((a - b).abs().max()) for a, b in zip(deep, reference)]
    assert all(math.isfinite(value) for value in delta)
    before = delta[refresh_at - 1]

    def mean_labeled(values):
        return mean(x for x in values if x is not None)

    return dict(pair=pair, residual_score=residual_score, loss_by_scan=losses, core_flops_by_scan=costs,
                future_write_mask_matches=mask_matches, deep_max_abs_vs_residual=delta,
                immediate_refresh_retention=None if before == 0 else delta[refresh_at] / before,
                current_loss=losses[0], future_loss=mean_labeled(losses[1:]), window_loss=mean_labeled(losses),
                pre_refresh_loss=mean_labeled(losses[:refresh_at]), post_refresh_loss=mean_labeled(losses[refresh_at:])), deep


def probe(cache, trace, clip, scene, criterion):
    matched = 0
    for index in range(len(trace) - WINDOW + 1):
        window = trace[index:index + WINDOW]
        refresh_at = periodic_offset(window)
        if refresh_at is None:
            continue
        root = window[0]
        frame = cache.prepare(root.spatial, root.prior, root.meta)
        required = cache.mandatory(frame).cpu()
        base_sites, sites, pairs = group_bucket(cache, frame, required, CAP)
        if len(pairs) < 2:
            continue
        score = tile_features(frame)[..., 0]
        values = [sum(float(score[key]) for key in pair) for pair in pairs]
        residual_index = max(range(len(pairs)), key=values.__getitem__)
        residual, reference = evaluate_group(cache, window, pairs[residual_index], required, sites,
                                             values[residual_index], criterion, refresh_at)
        groups = []
        for i, pair in enumerate(pairs):
            evaluation = residual if i == residual_index else evaluate_group(
                cache, window, pair, required, sites, values[i], criterion, refresh_at, reference)[0]
            groups.append(evaluation)
        cost_match = len({tuple(group['core_flops_by_scan']) for group in groups}) == 1
        writes_match = all(group['future_write_mask_matches'] == [True] * (WINDOW - 2) for group in groups)
        comparable = cost_match and writes_match
        row = dict(clip=clip, scene=scene, frames=[scan.meta['frame'] for scan in window],
                   timestamps_us=[int(scan.meta['timestamp_us']) for scan in window],
                   refresh_reasons=[scan.refresh for scan in window], periodic_offset=refresh_at,
                   base_sites=base_sites, sites=sites, lattice_sites=required.numel(),
                   cost_match=cost_match, future_writes_match=writes_match, matched=comparable, groups=groups)
        if comparable:
            row['selected'] = dict(residual=residual_index,
                current_oracle=min(range(len(pairs)), key=lambda i: groups[i]['current_loss']),
                window_oracle=min(range(len(pairs)), key=lambda i: groups[i]['window_loss']))
            matched += 1
        yield row
        if matched >= PROBES_PER_CLIP:
            break


def summarize_refresh(windows):
    result = summarize(windows)
    matched = [row for row in windows if row['matched']]
    if not matched:
        return result
    pairs = [(row['periodic_offset'], g) for row in matched for i, g in enumerate(row['groups'])
             if i != row['selected']['residual']]
    refresh_max = [g['deep_max_abs_vs_residual'][i] for i, g in pairs]
    before_max = [g['deep_max_abs_vs_residual'][i - 1] for i, g in pairs]
    retained = [g['immediate_refresh_retention'] for _, g in pairs if g['immediate_refresh_retention'] is not None]
    result['deep_difference'] = dict(non_residual_branches=len(pairs),
        pre_refresh_nonzero=sum(value > 0 for value in before_max), refresh_nonzero=sum(value > 0 for value in refresh_max),
        refresh_above_epsilon=sum(value > EPSILON for value in refresh_max),
        max_abs_pre_refresh=max(before_max), max_abs_at_refresh=max(refresh_max),
        mean_retention_ratio=mean(retained) if retained else None)
    for policy, losses in result['mean_loss'].items():
        chosen = [(row['periodic_offset'], row['groups'][row['selected'][policy]]) for row in matched]
        losses.update({phase:mean(group[phase] for _, group in chosen) for phase in PHASES})
        result['deep_difference'][policy] = dict(
            windows_state_diff_at_refresh=sum(group['deep_max_abs_vs_residual'][i] > EPSILON for i, group in chosen),
            max_abs_at_refresh=max(group['deep_max_abs_vs_residual'][i] for i, group in chosen))
    for left, right in (('residual', 'current_oracle'), ('residual', 'window_oracle'), ('current_oracle', 'window_oracle')):
        contrast = result['gains'][right + '_over_' + left]
        for phase in PHASES:
            gains = [row['groups'][row['selected'][left]][phase] - row['groups'][row['selected'][right]][phase]
                     for row in matched]
            contrast[phase] = dict(mean=mean(gains), improves=sum(x > EPSILON for x in gains),
                ties=sum(abs(x) <= EPSILON for x in gains), worsens=sum(x < -EPSILON for x in gains))
    return result


def run(args, out):
    plan_hash = digest(Path(args.plan))
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train' and len(plan['scenes']) == len(plan['clips'])
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == plan['split']
    source.period, source.max_age = 5, .55
    cache = DeepBlockCache(source, TILE, 'light')
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    original = model_digest(source)
    write_json(out / 'contract.json', dict(plan_sha256=plan_hash, source_sha256=SOURCE_SHA,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), torch=torch.__version__,
        device=torch.cuda.get_device_name(), period=source.period, max_age=source.max_age, tile=TILE, cap=CAP,
        window=WINDOW, max_matched_windows_per_clip=PROBES_PER_CLIP, delta_epsilon=EPSILON,
        eligibility='Ordinary labeled t, exactly one scheduled period refresh at t+1 or t+2, at least one label at or after it',
        objective='One equal-cost candidate write at t from common causal geometry history. Common future masks on ordinary scans and unmodified periodic refresh. Label oracles offline only.',
        scope='Training-side frozen detector loss and core Conv/ConvTranspose FLOPs. No training, AP, latency or power.'))
    all_windows, per_clip = [], []
    with torch.no_grad(), (out / 'frames.jsonl').open('x', buffering=1) as raw, \
            (out / 'trace.jsonl').open('x', buffering=1) as trace_log:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            trace = fixed_history(cache, ids, stream, trace_log, clip)
            scene, segment = plan['scenes'][clip]
            assert all(scan.meta['segment'] == segment for scan in trace)
            windows = list(probe(cache, trace, clip, scene, criterion))
            for window in windows:
                raw.write(json.dumps(window, allow_nan=False) + '\n')
            per_clip.append(dict(clip=clip, scene=scene, scans=len(trace), **summarize_refresh(windows)))
            all_windows.extend(windows)
            del trace
            write_json(out / 'status.json', dict(status='RUNNING', clips=clip + 1, windows=len(all_windows)))
    assert model_digest(source) == original and digest(Path(args.plan)) == plan_hash
    summary = summarize_refresh(all_windows)
    per_offset = {str(i):summarize_refresh([row for row in all_windows if row['periodic_offset'] == i])
                  for i in range(1, WINDOW)}
    write_json(out / 'report.json', dict(status='COMPLETED', weights_unchanged=True,
        parameters=sum(p.numel() for p in source.parameters()), scans=sum(map(len, plan['clips'])),
        matched_clips=sum(row['matched_windows'] > 0 for row in per_clip), per_clip=per_clip, per_offset=per_offset,
        ap50=None, ap70=None, model_step_ms_mean=None, **summary))
    write_json(out / 'status.json', dict(status='COMPLETED', matched_windows=summary['matched_windows']))


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

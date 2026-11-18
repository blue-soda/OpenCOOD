"""Test two aligned past residual features on archived equal-cost loss windows."""
import argparse
from collections import defaultdict
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_state_cache import relative_change_score
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, FEATURES, TILE, checked_plan, clip_streams
from opencood.tools.fit_int_deep_block_window_rank import (
    EPSILON, OBJECTIVES, REGULARIZATION, comparison_stats, leave_scene_out, read_jsonl, reconstruct_window,
)
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build
from opencood.tools.probe_int_deep_block_window import METRICS, WINDOW, fixed_history
from opencood.tools.run_int_ego import write_json


PAST_FEATURES = ('aligned_previous_log_relative_change', 'overlap_minimum_log_relative_change')
BASE_MODES = ('residual', *OBJECTIVES)
PAST_MODES = tuple('history_' + name for name in OBJECTIVES)


def overlap_features(current, aligned, overlap, tile):
    def pool(value):
        return F.avg_pool2d(value, tile, tile, ceil_mode=True, count_include_pad=False)

    support = overlap.to(current.dtype)
    fraction = pool(support)
    normalizer = torch.where(fraction > 0, fraction, torch.ones_like(fraction))
    values = (aligned, torch.minimum(aligned, current))
    return torch.cat([pool(value * support) / normalizer for value in values], 1), fraction


def history_features(cache, trace, clip, raw):
    previous = None
    result = {}
    for scan in trace:
        meta = scan.meta
        row = dict(frame=meta['frame'], clip=clip, refresh=scan.refresh, previous_frame=None,
                   dt_s=None, current_valid_cells=0, overlap_cells=0, lattice_cells=0,
                   history_bytes=0, status='refresh')
        if scan.refresh:
            previous = None
        else:
            frame = cache.prepare(scan.spatial, scan.prior, meta)
            size = frame.aligned.shape[-2:]
            current = F.adaptive_avg_pool2d(frame.coarse, size)
            prior = F.adaptive_avg_pool2d(frame.old_coarse, size)
            magnitude = relative_change_score(current, prior, 1).log1p()
            aligned = torch.zeros_like(magnitude)
            overlap = torch.zeros_like(frame.valid)
            row.update(dt_s=frame.dt, current_valid_cells=int(frame.valid.sum()), lattice_cells=frame.valid.numel(),
                       status='previous_refresh')
            if previous is not None:
                past_magnitude, past_valid, past_meta = previous
                assert past_meta['segment'] == meta['segment']
                dt = (int(meta['timestamp_us']) - int(past_meta['timestamp_us'])) / 1e6
                assert 0 < dt <= .25 and abs(dt - frame.dt) <= 1e-12
                bounds = block_lidar_range(cache.source.bounds, scan.spatial.shape[-2:], size, 8)
                packed = torch.cat((past_magnitude, past_valid.to(magnitude.dtype)), 1)
                warped, geometric_valid = warp_bev(packed, past_meta['pose'], meta['pose'], bounds)
                aligned = warped[:, :1]
                overlap = frame.valid & geometric_valid & (warped[:, 1:] > .5)
                row.update(previous_frame=past_meta['frame'], status='aligned_previous_ordinary',
                           overlap_cells=int(overlap.sum()))
            pooled, support = overlap_features(magnitude, aligned, overlap, cache.tile)
            row['history_bytes'] = magnitude.numel() * magnitude.element_size() + frame.valid.numel() * frame.valid.element_size()
            result[meta['frame']] = (pooled[0].permute(1, 2, 0).cpu().numpy().astype(np.float64),
                                     support[0, 0].cpu().numpy().astype(np.float64), row.copy())
            previous = magnitude, frame.valid, meta
        raw.write(json.dumps(row, allow_nan=False) + '\n')
    return result


def checked_reference(args):
    inputs = {k: digest(Path(getattr(args, k))) for k in ('plan', 'trace', 'windows', 'probe_contract')}
    ref = Path(args.reference)
    paths = {name: ref / name for name in ('contract.json', 'report.json', 'features.jsonl', 'predictions.jsonl')}
    hashes = {name: digest(path) for name, path in paths.items()}
    old = json.loads(paths['contract.json'].read_text())
    report = json.loads(paths['report.json'].read_text())
    probe = json.loads(Path(args.probe_contract).read_text())
    assert old['inputs_sha256'] == report['inputs_sha256'] == inputs
    assert old['features'] == list(FEATURES) and old['regularization'] == REGULARIZATION
    assert old['objectives'] == OBJECTIVES and old['tie_epsilon'] == EPSILON
    assert old['source_probe_code_commit'] == probe['code_commit']
    assert old['weight_sha256'] == probe['source_sha256'] == SOURCE_SHA
    assert probe['plan_sha256'] == inputs['plan']
    assert (probe['tile'], probe['cap'], probe['period'], probe['max_age'], probe['window']) == (TILE, CAP, 5, .55, WINDOW)
    assert report['status'] == 'COMPLETED'
    assert report['outputs_sha256'] == {name: hashes[name] for name in ('features.jsonl', 'predictions.jsonl')}
    contract = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        source_rank_code_commit=old['code_commit'], source_probe_code_commit=probe['code_commit'], inputs_sha256=inputs,
        rank_reference_sha256=hashes, weight_sha256=SOURCE_SHA, features=FEATURES, history_features=PAST_FEATURES,
        objectives=OBJECTIVES, regularization=REGULARIZATION, tie_epsilon=EPSILON, no_hyperparameter_search=True,
        fold='Hold out entire scenes; use same samples, centering, training-only scaling and penalty across objectives',
        history='Use immediately preceding ordinary scan only. Nearest-neighbor metric warp of per-cell log1p relative residual and validity to t; pool over valid overlap, with no overlap mapped to zero.',
        reset='Clear history on every full refresh and clip boundary. Use real dt to verify causal adjacency; no decay or time-constant search.',
        scope='Matched frozen detector losses on training-side windows. Future labeled outcomes used as targets only; no held-out validation AP, timing or energy.')
    return contract, paths, report


def replay(cache, plan, rows, cfg, trace, windows, archived_samples, out):
    trace_by_clip, windows_by_clip = defaultdict(list), defaultdict(list)
    archived_by_frames = {tuple(row['frames']): row for row in archived_samples}
    assert len(archived_by_frames) == len(archived_samples)
    for row in trace:
        trace_by_clip[row['clip']].append(row)
    for row in windows:
        assert row['matched'] == (row['cost_match'] and row['future_writes_match'])
        if row['matched']:
            windows_by_clip[row['clip']].append(row)
    baseline, extended = [], []
    with (out / 'features.jsonl').open('x', buffering=1) as raw, \
            (out / 'history_diagnostics.jsonl').open('x', buffering=1) as diagnostics:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            journal = StringIO()
            neutral = fixed_history(cache, ids, stream, journal, clip)
            assert [json.loads(line) for line in journal.getvalue().splitlines()] == trace_by_clip.pop(clip), clip
            assert all(scan.meta['segment'] == plan['scenes'][clip][1] for scan in neutral), clip
            past = history_features(cache, neutral, clip, diagnostics)
            for row in windows_by_clip.pop(clip, []):
                sample = reconstruct_window(cache, neutral, row, plan['scenes'][clip][0])
                key = tuple(sample['frames'])
                assert json.loads(json.dumps(sample)) == archived_by_frames.pop(key), key
                maps, support, history = past[key[0]]
                paired = np.stack([maps[a] + maps[b] for a, b in sample['pairs']])
                assert paired.shape == (len(sample['pairs']), len(PAST_FEATURES)) and np.isfinite(paired).all()
                vector = np.column_stack((sample['x'], paired))
                extended_sample = dict(sample, x=vector.tolist(), history=history,
                    overlap_fractions_by_pair=[[float(support[a]), float(support[b])] for a, b in sample['pairs']])
                raw.write(json.dumps(extended_sample, allow_nan=False) + '\n')
                baseline.append(sample)
                extended.append(extended_sample)
            del neutral
            write_json(out / 'status.json', dict(status='RUNNING', phase='replaying', clips=clip + 1, matched=len(baseline)))
    assert not (archived_by_frames or trace_by_clip or windows_by_clip)
    assert {row['scene'] for row in baseline} == {entry[0] for entry in plan['scenes']}
    return baseline, extended


def fit_extended(train):
    windows = [np.asarray(row['x'], dtype=np.float64) for row in train]
    centered = np.concatenate([window - window.mean(axis=0) for window in windows])
    scale = np.maximum(centered.std(axis=0), 1e-6)
    features = centered / scale
    count, width = features.shape
    targets = np.concatenate([np.column_stack([row['targets'][mode] for mode in OBJECTIVES]) for row in train])
    system = features.T @ features / count + REGULARIZATION * np.eye(width)
    weights = np.linalg.solve(system, features.T @ targets / count) / scale[:, None]
    assert np.isfinite(weights).all()
    return weights, scale, count


def crossfit(baseline, extended, rank_predictions, report):
    folds, predictions = leave_scene_out(baseline)
    assert json.loads(json.dumps(predictions)) == rank_predictions
    archived_folds = {fold['held_scene']: fold for fold in report['folds']}
    by_frames = {tuple(row['frames']): row for row in extended}
    max_baseline_delta = 0.
    for fold in folds:
        scene = fold['held_scene']
        archived = archived_folds.pop(scene)
        for field in ('train_scenes', 'train_windows', 'train_candidates'):
            assert fold[field] == archived[field], scene
        np.testing.assert_allclose(fold['feature_scale'], archived['feature_scale'], rtol=1e-12, atol=1e-12)
        fresh_w = np.column_stack([fold['weights'][mode] for mode in OBJECTIVES])
        old_w = np.column_stack([archived['weights'][mode] for mode in OBJECTIVES])
        weights, scale, count = fit_extended([row for row in extended if row['scene'] != scene])
        assert fold['train_candidates'] == count
        fold['history_feature_scale'] = scale.tolist()
        fold['history_weights'] = {mode: weights[:, j].tolist() for j, mode in enumerate(PAST_MODES)}
        for prediction in (row for row in predictions if row['scene'] == scene):
            sample = by_frames.pop(tuple(prediction['frames']))
            x = np.asarray(sample['x'], dtype=np.float64)
            now_score = x[:, :len(FEATURES)] @ fresh_w
            old_score = x[:, :len(FEATURES)] @ old_w
            max_baseline_delta = max(max_baseline_delta, float(np.abs(now_score - old_score).max()))
            np.testing.assert_allclose(now_score, old_score, atol=1e-12, rtol=1e-12)
            scores = x @ weights
            for j, mode in enumerate(PAST_MODES):
                index = int(np.argmax(scores[:, j]))
                prediction['selected'][mode] = index
                prediction['selected_pairs'][mode] = sample['pairs'][index]
                prediction['losses'][mode] = {field: sample['losses'][field][index] for field in METRICS}
    assert not by_frames and not archived_folds
    return folds, predictions, max_baseline_delta


def summarize(predictions):
    modes = (*BASE_MODES, *PAST_MODES)
    mean_loss = {mode:{field: mean(row['losses'][mode][field] for row in predictions) for field in METRICS}
                 for mode in modes}
    by_scene = defaultdict(list)
    for row in predictions:
        by_scene[row['scene']].append(row)
    pairs = [('residual', mode) for mode in modes if mode != 'residual']
    pairs += list(zip(OBJECTIVES, PAST_MODES)) + [tuple(OBJECTIVES), PAST_MODES]
    comparisons = {}
    for left, right in pairs:
        result = {}
        for field in METRICS:
            gains = {scene: [r['losses'][left][field] - r['losses'][right][field] for r in rows]
                     for scene, rows in by_scene.items()}
            scenes = {scene: mean(values) for scene, values in gains.items()}
            result[field] = dict(by_window=comparison_stats([v for values in gains.values() for v in values]),
                                 by_scene=comparison_stats(list(scenes.values())), scene_mean_gains=scenes)
        result['changed_choices'] = sum(row['selected'][left] != row['selected'][right] for row in predictions)
        comparisons[right + '_over_' + left] = result
    return dict(mean_loss=mean_loss, comparisons=comparisons, windows_per_scene={s:len(rows) for s, rows in by_scene.items()})


def run(args, out):
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train' and len({r[0] for r in plan['scenes']}) == len(plan['clips']) == 12
    contract, ref, prior_report = checked_reference(args)
    trace, windows = read_jsonl(args.trace), read_jsonl(args.windows)
    archived_samples = read_jsonl(ref['features.jsonl'])
    archived_predictions = read_jsonl(ref['predictions.jsonl'])
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == plan['split']
    source.period, source.max_age = 5, .55
    cache = DeepBlockCache(source, TILE, 'light')
    original = model_digest(source)
    write_json(out / 'contract.json', dict(contract, torch=torch.__version__, device=torch.cuda.get_device_name()))
    with torch.no_grad():
        baseline, extended = replay(cache, plan, rows, cfg, trace, windows, archived_samples, out)
    assert model_digest(source) == original
    assert all(digest(Path(getattr(args, k))) == h for k, h in contract['inputs_sha256'].items())
    assert all(digest(ref[k]) == h for k, h in contract['rank_reference_sha256'].items())
    folds, predictions, delta = crossfit(baseline, extended, archived_predictions, prior_report)
    with (out / 'predictions.jsonl').open('x') as raw:
        for row in predictions:
            raw.write(json.dumps(row, allow_nan=False) + '\n')
    diagnostics = read_jsonl(out / 'history_diagnostics.jsonl')
    parameters = sum(p.numel() for p in source.parameters())
    result = dict(status='COMPLETED', matched_windows=len(baseline), candidates=sum(len(r['pairs']) for r in baseline),
        frozen_weight_unchanged=True, r281_selections_and_losses_exact=True, r281_baseline_score_max_abs_delta=delta,
        parameters_by_mode=dict(residual=parameters, **{m:parameters + len(FEATURES) for m in OBJECTIVES},
                                **{m:parameters + len(FEATURES) + len(PAST_FEATURES) for m in PAST_MODES}),
        mean_core_flops_by_scan=[mean(r['verified_core_flops'][i] for r in baseline) for i in range(WINDOW)], folds=folds,
        history_state_bytes_max=max(row['history_bytes'] for row in diagnostics),
        windows_with_history=sum(r['history']['previous_frame'] is not None for r in extended),
        windows_with_overlap=sum(r['history']['overlap_cells'] > 0 for r in extended),
        ap50=None, ap70=None, model_step_ms_mean=None,
        outputs_sha256={name: digest(out / name) for name in ('features.jsonl', 'history_diagnostics.jsonl', 'predictions.jsonl')},
        **summarize(predictions))
    write_json(out / 'report.json', result)
    write_json(out / 'status.json', dict(status='COMPLETED', held_scenes=len(folds), matched_windows=len(baseline)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'trained', 'readout', 'output', 'plan', 'trace', 'windows', 'probe-contract', 'reference'):
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

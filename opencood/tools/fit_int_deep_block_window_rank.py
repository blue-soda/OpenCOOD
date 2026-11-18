"""Compare two causal rankers using held-scene, paired short-window loss."""
import argparse
from collections import defaultdict
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, FEATURES, TILE, checked_plan, clip_streams, tile_features
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket
from opencood.tools.probe_int_deep_block_window import METRICS, WINDOW, fixed_history
from opencood.tools.run_int_ego import write_json


OBJECTIVES = dict(current_ranker='current_loss', window_ranker='window_loss')
REGULARIZATION = .05
EPSILON = 1e-6


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def verify_group_cost(cache, window, pair, required, sites, expected):
    state = window[0].prior
    request = add_tiles(torch.zeros_like(required), pair, cache.tile).cuda()
    for offset, scan in enumerate(window):
        if offset:
            def fixed_request(frame, mandatory):
                combined = mandatory if scan.request is None else (mandatory | scan.request)
                assert torch.equal(combined, scan.write_mask), scan.meta['frame']
                return scan.request

            request = fixed_request
        (_, state, info), cost = measured_step(cache, scan.spatial, state, scan.meta, request)
        assert info['full_refresh'] is None and cost == expected[offset], scan.meta['frame']
        if offset == 0:
            assert abs(info['stage_conv_ratio'] - sites / required.numel()) <= 1e-12


def reconstruct_window(cache, trace, row, scene):
    by_frame = {scan.meta['frame']: i for i, scan in enumerate(trace)}
    start = by_frame[row['frames'][0]]
    window = trace[start:start + WINDOW]
    scans = [scan.meta['frame'] for scan in window]
    assert scans == row['frames'] and all(scan.refresh is None for scan in window), scans
    at = window[0]
    frame = cache.prepare(at.spatial, at.prior, at.meta)
    required = cache.mandatory(frame).cpu()
    base_sites, sites, pairs = group_bucket(cache, frame, required, CAP)
    assert (base_sites, sites, required.numel()) == (row['base_sites'], row['sites'], row['lattice_sites']), scans
    groups = row['groups']
    archived_pairs = [tuple(tuple(key) for key in g['pair']) for g in groups]
    assert pairs == archived_pairs, scans
    features = tile_features(frame)
    vectors = np.stack([features[a] + features[b] for a, b in pairs])
    residual = [g['residual_score'] for g in groups]
    assert np.array_equal(vectors[:, 0], residual), (scans, vectors[:, 0], residual)
    selected = int(np.argmax(vectors[:, 0]))
    assert selected == row['selected']['residual'], scans
    cost = groups[0]['core_flops_by_scan']
    for group in groups:
        assert group['core_flops_by_scan'] == cost and group['future_write_mask_matches'] == [True] * (WINDOW - 1), scans
        labels = group['loss_by_scan']
        assert [v is not None for v in labels] == [scan.target is not None for scan in window], scans
        assert labels[0] == group['current_loss'], scans
        assert abs(mean(v for v in labels[1:] if v is not None) - group['future_loss']) < 1e-12, scans
        assert abs(mean(v for v in labels if v is not None) - group['window_loss']) < 1e-12, scans
    verify_group_cost(cache, window, pairs[selected], required, sites, cost)
    losses = {metric: np.array([g[metric] for g in groups], dtype=np.float64) for metric in METRICS}
    targets = {mode: (losses[metric].mean() - losses[metric]).tolist() for mode, metric in OBJECTIVES.items()}
    assert np.isfinite(vectors).all() and all(np.isfinite(values).all() for values in losses.values()), scans
    return dict(scene=scene, clip=row['clip'], frames=scans, pairs=pairs, x=vectors.tolist(),
                targets=targets, losses={k: v.tolist() for k, v in losses.items()}, verified_core_flops=cost)


def collect_features(cache, plan, rows, cfg, archived_trace, archived_windows, out):
    traces, windows = defaultdict(list), defaultdict(list)
    for row in archived_trace:
        traces[row['clip']].append(row)
    for row in archived_windows:
        assert row['matched'] == (row['cost_match'] and row['future_writes_match']), row['frames']
        if row['matched']:
            windows[row['clip']].append(row)
    samples = []
    with (out / 'features.jsonl').open('x', buffering=1) as raw:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            journal = StringIO()
            trace = fixed_history(cache, ids, stream, journal, clip)
            assert [json.loads(line) for line in journal.getvalue().splitlines()] == traces.pop(clip), clip
            assert all(scan.meta['segment'] == plan['scenes'][clip][1] for scan in trace), clip
            for row in windows.pop(clip, []):
                sample = reconstruct_window(cache, trace, row, plan['scenes'][clip][0])
                raw.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.append(sample)
            del trace
            write_json(out / 'status.json', dict(status='RUNNING', phase='replaying', clips=clip + 1, matched=len(samples)))
    assert not traces and not windows
    assert len(samples) == sum(row['matched'] for row in archived_windows)
    assert {s['scene'] for s in samples} == {r[0] for r in plan['scenes']}
    return samples


def fit(train):
    windows = [np.asarray(row['x'], dtype=np.float64) for row in train]
    centered = np.concatenate([window - window.mean(axis=0) for window in windows])
    scale = np.maximum(centered.std(axis=0), 1e-6)
    features = centered / scale
    targets = np.concatenate([np.column_stack([row['targets'][mode] for mode in OBJECTIVES]) for row in train])
    count = len(features)
    system = features.T @ features / count + REGULARIZATION * np.eye(len(FEATURES))
    weights = np.linalg.solve(system, features.T @ targets / count) / scale[:, None]
    assert np.isfinite(weights).all()
    return weights, scale, count


def leave_scene_out(samples):
    scenes = sorted({r['scene'] for r in samples})
    folds, predictions = [], []
    for scene in scenes:
        train = [row for row in samples if row['scene'] != scene]
        held = [row for row in samples if row['scene'] == scene]
        weights, scale, count = fit(train)
        folds.append(dict(held_scene=scene, train_scenes=[s for s in scenes if s != scene],
            train_windows=len(train), train_candidates=count, feature_scale=scale.tolist(),
            weights={mode: weights[:, j].tolist() for j, mode in enumerate(OBJECTIVES)}))
        for row in held:
            x = np.asarray(row['x'], dtype=np.float64)
            scores = x @ weights
            selected = dict(residual=int(np.argmax(x[:, 0])))
            selected.update({mode: int(np.argmax(scores[:, j])) for j, mode in enumerate(OBJECTIVES)})
            losses = {mode:{field: row['losses'][field][index] for field in METRICS}
                      for mode, index in selected.items()}
            predictions.append(dict(scene=scene, clip=row['clip'], frames=row['frames'], selected=selected,
                selected_pairs={mode: row['pairs'][index] for mode, index in selected.items()}, losses=losses))
    assert len(predictions) == len(samples)
    return folds, predictions


def comparison_stats(deltas):
    return dict(mean=mean(deltas), wins=sum(v > EPSILON for v in deltas),
                ties=sum(abs(v) <= EPSILON for v in deltas), losses=sum(v < -EPSILON for v in deltas))


def summary(predictions):
    modes = ('residual', *OBJECTIVES)
    mean_losses = {m:{f:mean(r['losses'][m][f] for r in predictions) for f in METRICS} for m in modes}
    per_scene = defaultdict(list)
    for row in predictions:
        per_scene[row['scene']].append(row)
    comparisons = {}
    for left, right in (('residual', 'current_ranker'), ('residual', 'window_ranker'), ('current_ranker', 'window_ranker')):
        comparison = {}
        for field in METRICS:
            by_scene = {scene:mean(r['losses'][left][field] - r['losses'][right][field] for r in rows)
                        for scene, rows in per_scene.items()}
            gains = [r['losses'][left][field] - r['losses'][right][field] for r in predictions]
            comparison[field] = dict(by_window=comparison_stats(gains),
                by_scene=comparison_stats(list(by_scene.values())), scene_mean_gains=by_scene)
        comparison['changed_choices'] = sum(row['selected'][left] != row['selected'][right] for row in predictions)
        comparisons[right + '_over_' + left] = comparison
    return dict(mean_loss=mean_losses, comparisons=comparisons, windows_per_scene={s:len(v) for s, v in per_scene.items()})


def run(args, out):
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train' and len(plan['scenes']) == len(plan['clips']) == 12
    assert len({pair[0] for pair in plan['scenes']}) == len(plan['scenes'])
    inputs = {name: digest(Path(getattr(args, name))) for name in ('plan', 'trace', 'windows', 'probe_contract')}
    prior = json.loads(Path(args.probe_contract).read_text())
    assert prior['plan_sha256'] == inputs['plan'] and prior['source_sha256'] == SOURCE_SHA
    assert (prior['tile'], prior['cap'], prior['period'], prior['max_age'], prior['window']) == (TILE, CAP, 5, .55, WINDOW)
    trace, windows = read_jsonl(args.trace), read_jsonl(args.windows)
    assert len(trace) == sum(map(len, plan['clips']))
    assert len({tuple(row['frames']) for row in windows}) == len(windows)
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == plan['split']
    source.period, source.max_age = 5, .55
    cache = DeepBlockCache(source, TILE, 'light')
    before = model_digest(source)
    write_json(out / 'contract.json', dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        source_probe_code_commit=prior['code_commit'], inputs_sha256=inputs, weight_sha256=SOURCE_SHA,
        torch=torch.__version__, device=torch.cuda.get_device_name(), features=FEATURES, objectives=OBJECTIVES,
        fold='Leave one entire scene out; all matches from every other scene used for training only',
        scores='Only five causal t-frame pair features; negative losses are target-only, centered within windows',
        standardization='Center each feature within each t group, normalize by training-fold std (floor 1e-6)',
        regularization=REGULARIZATION, tie_epsilon=EPSILON, no_hyperparameter_search=True))
    with torch.no_grad():
        samples = collect_features(cache, plan, rows, cfg, trace, windows, out)
    assert model_digest(source) == before
    assert all(inputs[name] == digest(Path(getattr(args, name))) for name in inputs)
    folds, predictions = leave_scene_out(samples)
    with (out / 'predictions.jsonl').open('x') as raw:
        for row in predictions:
            raw.write(json.dumps(row, allow_nan=False) + '\n')
    report = dict(status='COMPLETED', matched_windows=len(samples), source_windows=len(windows),
        candidates=sum(len(r['pairs']) for r in samples), weight_unchanged=True, inputs_sha256=inputs,
        outputs_sha256={name: digest(out / name) for name in ('features.jsonl', 'predictions.jsonl')},
        parameters=sum(p.numel() for p in source.parameters()) + len(FEATURES), folds=folds,
        mean_core_flops_by_scan=[mean(r['verified_core_flops'][i] for r in samples) for i in range(WINDOW)],
        ap50=None, ap70=None, model_step_ms_mean=None, **summary(predictions))
    write_json(out / 'report.json', report)
    write_json(out / 'status.json', dict(status='COMPLETED', held_scenes=len(folds), matched_windows=len(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'trained', 'readout', 'output', 'plan', 'trace', 'windows', 'probe-contract'):
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

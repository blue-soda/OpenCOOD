"""Frozen P4/P5 deep-cache replay and label-only equal-work block probe."""
import argparse
from collections import Counter, defaultdict
import functools
import json
import math
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_state_cache import relative_change_score
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector, reset_reason
from opencood.models.sub_modules.int_pyramid_regions import plan_coalesced_refresh
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json


SOURCE_SHA = {
    'checkpoint': 'adac8d9616c933f74fcdad1ac8e112bb3ec82b82e73236563a76c576eca209bb',
    'trained': '32002d5c21231cc3ef44fbe97a013dc8f18c5f83703070c2ee14355ebf0a8c78',
    'readout': '375c8ebf981b27f1353c2eeae1690e48d5c7cabeea6cfaea5b1f10fc1e34f939',
}
ATOL = 1e-4
RTOL = 1e-4


def build(a):
    for name, expected in SOURCE_SHA.items():
        assert digest(Path(getattr(a, name))) == expected, name
    a.mode = 'gru'
    a.stage = 'calibrate' if a.kind == 'quality' else 'full'
    a.content_calibration = None
    teacher, cfg, rows, provenance = setup(a)
    base = PyramidStateDetector(teacher, cfg).cuda().eval()
    saved = torch.load(a.checkpoint, map_location='cpu')
    assert saved['contract']['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract'] == provenance['contract']
    base.load_state_dict(saved['model'])
    source = ProtectedMultiscaleRefreshDetector(base, True).cuda().eval()
    load_targets(source, Path(a.trained), SOURCE_SHA['trained'])
    source.core.readout, _ = compact_readout(base.core.readout, 128)
    source.core.readout.load_state_dict(torch.load(a.readout, map_location='cpu')['readout'])
    return source.eval().requires_grad_(False), cfg, rows, provenance


def spatial_input(model, ego):
    lidar = ego['processed_lidar']
    if not lidar['voxel_features'].shape[0]:
        nx, ny, _ = model.scatter.model_cfg['grid_size']
        return model.core.cls_head.weight.new_zeros(1, 64, int(ny), int(nx))
    batch = {k: lidar[k] for k in ('voxel_features', 'voxel_coords', 'voxel_num_points')}
    return model.scatter(model.pillar_vfe(dict(batch, batch_size=1)))['spatial_features']


def masks_by_tile(required, tile):
    """Yield optional tiles in stable row-major order."""
    h, w = required.shape[-2:]
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if not bool(required[..., y:y+tile, x:x+tile].any()):
                yield y // tile, x // tile


def add_tiles(required, keys, tile):
    result = required.clone()
    for y, x in keys:
        result[..., y*tile:(y+1)*tile, x*tile:(x+1)*tile] = True
    return result


def group_bucket(cache, frame, required, cap):
    """Pick a metadata-only bucket of adjacent pairs at exactly equal planned work."""
    optional = list(masks_by_tile(required, cache.tile))
    allowed = set(optional)
    work = functools.partial(plan_coalesced_refresh, input_shape=frame.coarse.shape[-2:],
                             radius=cache.stage.radius, tile=cache.tile)
    base_work = work(required)[2]
    cells = required.numel()
    candidates = defaultdict(list)
    for y, x in optional:
        for other in ((y+1, x), (y, x+1)):
            if other not in allowed:
                continue
            pair = ((y, x), other)
            sites = work(add_tiles(required, pair, cache.tile))[2]
            if base_work < sites <= cap * cells:
                candidates[sites].append(pair)
    if not candidates:
        return base_work, None, []
    sites = max(candidates, key=lambda n: (len(candidates[n]), n))
    return base_work, sites, candidates[sites]


def max_delta(reference, actual, keys):
    maximum = 0.
    for key in keys:
        left, right = reference[key], actual[key]
        maximum = max(maximum, float((left - right).abs().max()))
        if not torch.allclose(left, right, atol=ATOL, rtol=RTOL):
            raise AssertionError(('same-mask mismatch', key, maximum))
    return maximum


def geometry(cache, spatial, meta, states, index):
    source = cache.source
    for period in (4, 5):
        source.period, source.max_age = period, period * .1 + .05
        (pred, states[period], info), flops = measured_step(source, spatial, states[period], meta)
        assert all(bool(torch.isfinite(v).all()) for v in pred.values())
        states[(period, 'trace')] = dict(flops=flops, refresh=info['full_refresh'], cache_bytes=info['cache_bytes'])
        if period == 4:
            p4_prediction = pred
    source.period, source.max_age = 4, .45
    mask = None
    if cache.refresh_reason(states['local'], meta) is None:
        frame = cache.prepare(spatial, states['local'], meta)
        required = cache.mandatory(frame).cpu()
        keys = list(masks_by_tile(required, cache.tile))
        if keys:
            mask = add_tiles(torch.zeros_like(required), (keys[index * 7 % len(keys)],), cache.tile).cuda()
    (pred, states['local'], info), local_flops = measured_step(cache, spatial, states['local'], meta, mask)
    (ref, states['ref'], _), ref_flops = measured_step(Reference(cache), spatial, states['ref'], meta, mask)
    prediction_delta = max_delta(ref, pred, pred)
    state_delta = max_delta(states['ref'], states['local'], (*source.state_keys, 'deep_age'))
    assert local_flops <= ref_flops
    assert info['full_refresh'] == states[(4, 'trace')]['refresh']
    record = dict(p4=states[(4, 'trace')], p5=states[(5, 'trace')], local=info,
                  local_flops=local_flops, same_mask_dense_flops=ref_flops,
                  prediction_max_abs=prediction_delta, state_max_abs=state_delta)
    if cache.operator == 'light':
        all_mask = torch.ones_like(states['local']['deep_age'], dtype=torch.bool)
        (all_pred, states['all'], all_info), all_flops = measured_step(cache, spatial, states['all'], meta, all_mask)
        record.update(p4_all_write_prediction_max_abs=max_delta(p4_prediction, all_pred, p4_prediction),
                      p4_all_write_state_max_abs=max_delta(states[4], states['all'], source.state_keys),
                      p4_all_write_flops=all_flops)
        assert all_flops == states[(4, 'trace')]['flops']
        assert all_info['full_refresh'] == states[(4, 'trace')]['refresh']
        assert local_flops <= all_flops
    return record


class Reference:
    def __init__(self, cache):
        self.cache, self.core = cache, cache.core

    def step_spatial(self, spatial, state, meta, requests=None):
        return self.cache.step_spatial(spatial, state, meta, requests, dense_reference=True)


def quality_probe(cache, spatial, meta, ego, state, criterion, required, frame, sites, pairs):
    source = cache.source
    candidate = cache.stage(cache.pack(frame))
    target = ego['label_dict']

    def loss(pred):
        value = float(criterion(pred, target))
        if not math.isfinite(value):
            raise ValueError('Non-finite detection loss')
        return value

    def interventions(keys):
        write = add_tiles(required, keys, cache.tile).cuda()
        return cache.predict(frame, torch.where(write, candidate, frame.aligned))

    previous = interventions(())
    baseline = loss(previous)
    cache.core.deep_update.delta_t = frame.dt
    light = cache.core.deep_update(cache.core.deep_light(frame.coarse), frame.aligned, frame.valid)
    full_state = dict(state, since_refresh=source.period - 1)
    refreshed, _, _ = source.step_spatial(spatial, full_state, meta)
    row = dict(mandatory_only=baseline, ordinary_light=loss(cache.predict(frame, light)), full_refresh=loss(refreshed))
    if cache.operator == 'heavy':
        row['heavy_light_coarse'] = loss(cache.predict(frame, candidate))
    else:
        (p4, _, _), p4_flops = measured_step(source, spatial, state, meta)
        max_delta(p4, cache.predict(frame, candidate), p4)
        row['p4_ordinary_core_flops'] = p4_flops
    if len(pairs) < 2:
        return row

    deep_shape = frame.aligned.shape[-2:]
    current = F.adaptive_avg_pool2d(frame.coarse, deep_shape)
    old = F.adaptive_avg_pool2d(frame.old_coarse, deep_shape)
    residual = relative_change_score(current, old, cache.tile)[0, 0].cpu()
    age = F.avg_pool2d(frame.age, cache.tile, cache.tile, ceil_mode=True, count_include_pad=False)[0, 0].cpu()
    foreground = F.adaptive_max_pool2d(previous['psm'].amax(1, keepdim=True), deep_shape)
    foreground = F.max_pool2d(foreground, cache.tile, cache.tile, ceil_mode=True)[0, 0].cpu()
    singles = {key: baseline - loss(interventions((key,))) for key in sorted({key for pair in pairs for key in pair})}
    outcomes = {pair: loss(interventions(pair)) for pair in pairs}
    ranks = dict(residual=residual, age=age, foreground=foreground, summed_single_benefit=singles)
    selected = {name: max(pairs, key=lambda pair: sum(float(score[key]) for key in pair)) for name, score in ranks.items()}
    selected['best_group_offline'] = min(pairs, key=outcomes.get)
    chosen = selected['summed_single_benefit']
    write = add_tiles(torch.zeros_like(required), chosen, cache.tile).cuda()
    (actual, actual_state, info), flops = measured_step(cache, spatial, state, meta, write)
    row.update(executed_model_core_flops=flops, checked_stage_ratio=info['stage_conv_ratio'])
    assert abs(info['stage_conv_ratio'] - sites / required.numel()) <= 1e-12
    if cache.operator == 'light':
        assert flops < row['p4_ordinary_core_flops']
    max_delta(interventions(chosen), actual, actual)
    expected = torch.where(add_tiles(required, chosen, cache.tile).cuda(), candidate, frame.aligned)
    max_delta({'deep': expected}, actual_state, ('deep',))
    row['candidates'] = [dict(tiles=pair, loss=outcomes[pair]) for pair in pairs]
    row['singles'] = [dict(tile=key, loss_reduction=gain) for key, gain in singles.items()]
    row['policies'] = {name: dict(tiles=pair, loss=outcomes[pair]) for name, pair in selected.items()}
    return row


def run(a, out):
    plan = json.loads(Path(a.plan).read_text())
    for path, expected in plan['sha256'].items():
        assert digest(Path(artifact_path(path))) == expected, path
    source, cfg, rows, provenance = build(a)
    assert provenance['split'] == plan['split']
    by_frame = {row['frame']: row for row in rows}
    cache = DeepBlockCache(source, a.tile, a.operator)
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval() if a.kind == 'quality' else None
    frozen = model_digest(source)
    shapes = [len(clip) for clip in plan['clips']]
    contract = dict(clips=plan['clips'], plan_sha256=digest(Path(a.plan)), kind=a.kind, tile=a.tile,
                    operator=a.operator,
                    cap=a.cap, max_probes=a.probes, probes_per_clip=a.probes_per_clip,
                    source_sha256=SOURCE_SHA, atol=ATOL, rtol=RTOL,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                    device=torch.cuda.get_device_name(), torch=torch.__version__, provenance=provenance,
                    quality_scope='Development labels; own geometry-only trace; single-frame loss, no AP or timing')
    write_json(out / 'contract.json', contract)
    records = []
    probes = 0
    with torch.no_grad(), (out / 'frames.jsonl').open('x', buffering=1) as log:
        for clip_index, ids in enumerate(plan['clips']):
            _, stream = loader([by_frame[f] for f in ids], cfg, 0, 303)
            states = {4: None, 5: None, 'local': None, 'ref': None, 'all': None}
            clip_probes = 0
            for index, sample in enumerate(stream):
                ego = to_device(sample['ego'], torch.device('cuda'))
                meta = sample['meta']
                assert meta['frame'] == ids[index]
                spatial = spatial_input(source, ego)
                row = dict(frame=meta['frame'], segment=meta['segment'], clip=clip_index,
                           supervised=bool(meta['supervised']))
                if a.kind == 'geometry':
                    row.update(geometry(cache, spatial, meta, states, index))
                else:
                    prior = states['local']
                    reason = cache.refresh_reason(prior, meta)
                    row['refresh'] = reason
                    write_mask = None
                    if reason is None:
                        frame = cache.prepare(spatial, prior, meta)
                        required = cache.mandatory(frame).cpu()
                        if meta['supervised'] and probes < a.probes and clip_probes < a.probes_per_clip:
                            base_sites, sites, pairs = group_bucket(cache, frame, required, a.cap)
                            row.update(base_sites=base_sites, budget_sites=sites, lattice_sites=required.numel(),
                                       equal_work_groups=len(pairs), **quality_probe(
                                           cache, spatial, meta, ego, prior, criterion, required, frame, sites, pairs))
                            probes += 1
                            clip_probes += 1
                        keys = list(masks_by_tile(required, cache.tile))
                        if keys:
                            write_mask = add_tiles(torch.zeros_like(required), (keys[(index * 7) % len(keys)],), cache.tile).cuda()
                    _, states['local'], row['trace'] = cache.step_spatial(spatial, prior, meta, write_mask)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                records.append(row)
                if len(records) % 8 == 0:
                    write_json(out / 'status.json', dict(status='RUNNING', scans=len(records), probes=probes))
    assert len(records) == sum(shapes) and model_digest(source) == frozen
    report = dict(status='COMPLETED', kind=a.kind, operator=a.operator, scans=len(records), clips=shapes, probes=probes,
                  parameters=sum(p.numel() for p in source.parameters()), ap50=None, ap70=None,
                  model_step_ms_mean=None, weights_unchanged=True)
    if a.kind == 'geometry':
        report.update(local_core_flops_mean=sum(r['local_flops'] for r in records)/len(records),
                      reference_core_flops_mean=sum(r['same_mask_dense_flops'] for r in records)/len(records),
                      prediction_max_abs=max(r['prediction_max_abs'] for r in records),
                      state_max_abs=max(r['state_max_abs'] for r in records),
                      refresh_reasons=dict(Counter(r['local']['full_refresh'] or 'ordinary' for r in records)))
        if a.operator == 'light':
            report.update(p4_all_write_prediction_max_abs=max(r['p4_all_write_prediction_max_abs'] for r in records),
                          p4_all_write_state_max_abs=max(r['p4_all_write_state_max_abs'] for r in records))
    else:
        matched = [r for r in records if 'policies' in r]
        report['matched_frames'] = len(matched)
        report['matched_clips'] = len({r['clip'] for r in matched})
        if matched:
            report['matched_loss_mean'] = {
                name: sum(r['policies'][name]['loss'] for r in matched)/len(matched)
                for name in matched[0]['policies']}
            report['checked_policy_core_flops_mean'] = sum(r['executed_model_core_flops'] for r in matched)/len(matched)
            if a.operator == 'light':
                report['matched_p4_ordinary_core_flops_mean'] = sum(r['p4_ordinary_core_flops'] for r in matched)/len(matched)
    write_json(out / 'report.json', report)
    write_json(out / 'status.json', dict(status='COMPLETED', scans=len(records), probes=probes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkpoint', 'trained', 'readout', 'plan', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--kind', required=True, choices=('geometry', 'quality'))
    parser.add_argument('--operator', choices=('heavy', 'light'), default='heavy')
    parser.add_argument('--tile', type=int, default=8)
    parser.add_argument('--cap', type=float, default=.9)
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--probes-per-clip', type=int, default=4)
    a = parser.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(status='RUNNING', phase='inputs'))
    try:
        run(a, out)
    except Exception as exc:
        write_json(out / 'status.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

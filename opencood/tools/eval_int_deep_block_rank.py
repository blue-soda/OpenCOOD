"""Fit a small causal block ranker, then evaluate frozen P5 cache policies."""
import argparse
import json
from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_local_state_cache import relative_change_score
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import (
    SOURCE_SHA, add_tiles, build, group_bucket, masks_by_tile, spatial_input,
)
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.utils import eval_utils


TILE = 8
CAP = .50
SEED = 303
FEATURES = ('log_relative_change', 'log_absolute_change', 'log_coarse_energy', 'log_deep_energy', 'age')


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    for p, expected in plan['sha256'].items():
        assert digest(Path(artifact_path(p))) == expected, p
    return plan


def tile_features(frame):
    size = frame.aligned.shape[-2:]
    current = F.adaptive_avg_pool2d(frame.coarse, size)
    old = F.adaptive_avg_pool2d(frame.old_coarse, size)

    def mean_tiles(x):
        return F.avg_pool2d(x, TILE, TILE, ceil_mode=True, count_include_pad=False)

    rel = relative_change_score(current, old, TILE)
    absolute = mean_tiles((current - old).abs().mean(1, keepdim=True))
    coarse = mean_tiles(current.square().mean(1, keepdim=True))
    deep = mean_tiles(frame.aligned.square().mean(1, keepdim=True))
    age = mean_tiles(frame.age)
    planes = [torch.log1p(x) for x in (rel, absolute, coarse, deep)] + [age]
    return torch.cat(planes, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)


def clip_streams(plan, rows, cfg):
    by_id = {row['frame']: row for row in rows}
    for clip, ids in enumerate(plan['clips']):
        stream_rows = [by_id[x] for x in ids]
        dataset, stream = loader(stream_rows, cfg, 0, SEED)
        yield clip, ids, dataset, stream


def gain_examples(cache, frame, required, pairs, criterion, target):
    candidate = cache.stage(cache.pack(frame))
    features = tile_features(frame)

    def loss(keys):
        mask = add_tiles(required, keys, TILE).to(frame.aligned.device)
        prediction = cache.predict(frame, torch.where(mask, candidate, frame.aligned))
        return float(criterion(prediction, target))

    reference = loss(())
    keys = sorted({k for pair in pairs for k in pair})
    gains = {k: reference - loss((k,)) for k in keys}
    assert all(np.isfinite(g) for g in gains.values())
    matrix = np.stack([features[a] + features[b] for a, b in pairs])
    targets = np.array([gains[a] + gains[b] for a, b in pairs])
    record = dict(tiles=[dict(key=k, features=features[k].tolist(), gain=gains[k]) for k in keys], pairs=pairs)
    return matrix - matrix.mean(0), targets - targets.mean(), record


def fit_ranker(cache, cfg, rows, plan, out):
    matrices, targets = [], []
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    with (out / 'training_examples.jsonl').open('x', buffering=1) as raw:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            state, counted = None, 0
            for index, sample in enumerate(stream):
                meta = sample['meta']
                assert meta['frame'] == ids[index]
                ego = to_device(sample['ego'], torch.device('cuda'))
                spatial = spatial_input(cache.source, ego)
                write_mask = None
                if cache.refresh_reason(state, meta) is None:
                    frame = cache.prepare(spatial, state, meta)
                    required = cache.mandatory(frame).cpu()
                    if meta['supervised'] and counted < 4:
                        _, sites, pairs = group_bucket(cache, frame, required, CAP)
                        if len(pairs) >= 2:
                            matrix, target, record = gain_examples(cache, frame, required, pairs, criterion, ego['label_dict'])
                            matrices.append(matrix)
                            targets.append(target)
                            raw.write(json.dumps(dict(clip=clip, frame=meta['frame'], sites=sites, **record), allow_nan=False)+'\n')
                            counted += 1
                    keys = list(masks_by_tile(required, TILE))
                    if keys:
                        write_mask = add_tiles(torch.zeros_like(required), (keys[index * 7 % len(keys)],), TILE).cuda()
                _, state, _ = cache.step_spatial(spatial, state, meta, write_mask)
            write_json(out / 'status.json', dict(status='RUNNING', phase='fit', clips=clip+1, examples=len(matrices)))
    assert len(matrices) >= 16, 'Insufficient eligible training frames'
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    scale = np.maximum(matrix.std(0), 1e-6)
    scaled = matrix / scale
    regularization = .05
    gram = scaled.T @ scaled / len(target)
    weights = np.linalg.solve(gram + regularization * np.eye(gram.shape[0]), scaled.T @ target / len(target)) / scale
    assert np.isfinite(weights).all()
    write_json(out / 'ranker.json', dict(weights=weights.tolist(), features=FEATURES, examples=len(matrices),
               pairs=len(target), regularization=regularization, frame_centered=True, training_scans=sum(map(len, plan['clips']))))
    return weights


class CausalPolicy:
    def __init__(self, cache, name, weights):
        self.cache, self.name, self.weights = cache, name, weights
        self.record = None

    def __call__(self, frame, required):
        mask = required.cpu()
        mandatory_sites, sites, pairs = group_bucket(self.cache, frame, mask, CAP)
        self.record = dict(mandatory_sites=mandatory_sites, planned_sites=mandatory_sites,
                           pairs=len(pairs), chosen=None)
        if not pairs:
            return None
        features = tile_features(frame)
        scores = features @ self.weights if self.name == 'learned' else features[..., 0 if self.name == 'residual' else 4]
        pair = max(pairs, key=lambda p: scores[p[0]] + scores[p[1]])
        self.record.update(chosen=pair, planned_sites=sites)
        return add_tiles(torch.zeros_like(mask), pair, TILE).to(required.device)


def policy_eval(cache, cfg, rows, plan, weights, out):
    modes = ('p5', 'residual', 'age', 'learned')
    policies = {name:CausalPolicy(cache, name, weights) for name in modes[1:]}
    stats = {n:{t:dict(tp=[], fp=[], score=[], gt=0) for t in (.5,.7)} for n in modes}
    records = []
    with (out / 'frames.jsonl').open('x', buffering=1) as raw:
        for clip, ids, dataset, stream in clip_streams(plan, rows, cfg):
            states = dict.fromkeys(modes)
            for index, sample in enumerate(stream):
                meta = sample['meta']
                assert meta['frame'] == ids[index]
                ego = to_device(sample['ego'], torch.device('cuda'))
                spatial = spatial_input(cache.source, ego)
                entry = dict(frame=meta['frame'], clip=clip, supervised=bool(meta['supervised']), arms={})
                batch = dict(ego=ego)
                gt = dataset.post.generate_gt_bbx(batch) if meta['supervised'] else None
                for mode in modes:
                    fn = policies.get(mode)
                    if fn is not None:
                        fn.record = None
                    detector = cache.source if mode == 'p5' else cache
                    (pred, states[mode], info), flops = measured_step(detector, spatial, states[mode], meta, fn)
                    assert all(bool(torch.isfinite(x).all()) for x in pred.values())
                    history = states[mode]
                    byte_count = sum(history[k].numel() * history[k].element_size() for k in cache.source.state_keys)
                    if 'deep_age' in history:
                        byte_count += history['deep_age'].numel() * history['deep_age'].element_size()
                    entry['arms'][mode] = dict(core_conv_flops=flops, refresh=info['full_refresh'], state_bytes=byte_count,
                                                schedule=None if fn is None else fn.record)
                    if meta['supervised']:
                        boxes, scores = dataset.post.post_process(batch, dict(ego=pred))
                        for t in (.5,.7):
                            eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[mode][t], t)
                reference = entry['arms']['p5']
                assert all(a['refresh'] == reference['refresh'] and a['core_conv_flops'] <= reference['core_conv_flops']
                           for a in entry['arms'].values())
                raw.write(json.dumps(entry, allow_nan=False)+'\n')
                records.append(entry)
                if len(records) % 20 == 0:
                    write_json(out / 'status.json', dict(status='RUNNING', phase='eval', scans=len(records)))
    write_json(out / 'ap_records.json', stats)
    write_json(out / 'sort_orders.json', dict(source_sha256=digest(out/'ap_records.json'), numpy_version=np.__version__,
        orders={mode:{str(t):np.argsort(-np.asarray(rec['score'])).tolist() for t,rec in stats[mode].items()} for mode in modes}))
    metrics = {}
    for mode in modes:
        steps = [r['arms'][mode] for r in records]
        metrics[mode] = dict(ap50=eval_utils.calculate_ap(stats[mode], .5)[0], ap70=eval_utils.calculate_ap(stats[mode], .7)[0],
            parameters=sum(p.numel() for p in cache.source.parameters())+(len(FEATURES) if mode=='learned' else 0),
            model_step_ms_mean=None, core_conv_flops_mean=sum(a['core_conv_flops'] for a in steps)/len(steps),
            state_bytes_max=max(a['state_bytes'] for a in steps), optional_updates=sum(bool(a['schedule'] and a['schedule']['chosen']) for a in steps),
            refresh=dict(Counter(a['refresh'] or 'ordinary' for a in steps)))
    matched = sum(len({r['arms'][name]['core_conv_flops'] for name in policies})==1 for r in records)
    return dict(metrics=metrics, scans=len(records), supervised=sum(r['supervised'] for r in records),
                gt=stats['p5'][.5]['gt'], exact_local_policy_cost_frames=matched)


def worker(a, out):
    train_plan, dev_plan = checked_plan(a.train_plan), checked_plan(a.dev_plan)
    assert train_plan['split']=='train' and dev_plan['split']=='val'
    assert {x[0] for x in train_plan['scenes']}.isdisjoint(x[0] for x in dev_plan['scenes'])
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               train_plan_sha256=digest(Path(a.train_plan)),dev_plan_sha256=digest(Path(a.dev_plan)),
               checkpoint_sha256=SOURCE_SHA,tile=TILE,core_execution_site_cap=CAP,period=5,max_age=.55,
               torch=torch.__version__,device=torch.cuda.get_device_name(),seed=SEED,
               scope='Fixed backbone. Train-only linear ranking from within-frame centered pair gains; frozen scorer and independently recurrent arms on held-out development scenes. No timing or energy.'))
    a.kind = 'quality'
    source, cfg, rows, _ = build(a)
    source.period, source.max_age = 5, .55
    before = model_digest(source)
    with torch.no_grad():
        weights = fit_ranker(DeepBlockCache(source, TILE, 'light'), cfg, rows, train_plan, out)
        assert model_digest(source)==before
        del source
        a.kind = 'geometry'
        source, cfg, rows, _ = build(a)
        source.period, source.max_age = 5, .55
        assert model_digest(source)==before
        result = policy_eval(DeepBlockCache(source, TILE, 'light'), cfg, rows, dev_plan, weights, out)
        assert model_digest(source)==before
    write_json(out/'report.json',dict(status='COMPLETED', backbone_unchanged=True, **result))
    write_json(out/'status.json',dict(status='COMPLETED', scans=result['scans']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('root','checkpoint','trained','readout','train-plan','dev-plan','output'):
        p.add_argument('--'+flag,required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    try:
        write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
        worker(a,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

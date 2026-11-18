"""Engineering gate for a differentiable, fixed-mask deep-cache training reference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import subprocess
from typing import NamedTuple

import numpy as np
import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, SEED, TILE, CausalPolicy, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.run_int_epochs import atomic_save


WINDOW = 3
TRAINED = ('core.deep_update.gate.2.weight', 'core.deep_update.gate.2.bias')
STATE_TENSORS = ('value', 'coarse', 'deep', 'deep_age')
LR = 1e-4
WEIGHT_DECAY = 1e-4


class Scan(NamedTuple):
    frame: str
    meta: dict
    spatial: torch.Tensor
    target: object
    refresh: object
    request: object
    written: object
    valid: object


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def tensor_digest(values, skip=()):
    result = hashlib.sha256()
    for name in sorted(values):
        if name not in skip:
            value = values[name].detach().cpu().contiguous()
            result.update(name.encode())
            result.update(value.numpy().tobytes())
    return result.hexdigest()


class DenseOrdinaryReference:
    """Only ordinary deep alignment, updates and their frozen readout retain a graph."""

    def __init__(self, source):
        self.cache = DeepBlockCache(source, TILE, 'light')
        self.core = source.core
        self.aligned = None

    def step_spatial(self, spatial, state, meta, requests=None):
        cache = self.cache
        self.aligned = None
        with torch.no_grad():
            if cache.refresh_reason(state, meta):
                return cache.step_spatial(spatial, state, meta, requests, dense_reference=True)
            frame = cache.prepare(spatial, state, meta)
            written = cache.mandatory(frame)
            optional = requests(frame, written) if callable(requests) else requests
            if optional is not None:
                written = written | optional
        bounds = block_lidar_range(cache.source.bounds, spatial.shape[-2:], state['deep'].shape[-2:], 8)
        aligned, valid = warp_bev(state['deep'], state['pose'], meta['pose'], bounds)
        assert torch.equal(valid, frame.valid) and torch.equal(aligned, frame.aligned), meta['frame']
        self.aligned = aligned
        candidate = cache.stage(cache.pack(frame._replace(aligned=aligned)))
        deep = torch.where(written, candidate, aligned)
        new = dict(value=frame.fine, coarse=frame.coarse, deep=deep,
                   deep_age=torch.where(written, torch.zeros_like(frame.age), frame.age),
                   pose=copy.deepcopy(meta['pose']), scene=meta['segment'], timestamp_us=int(meta['timestamp_us']),
                   refresh_timestamp_us=state['refresh_timestamp_us'], since_refresh=state['since_refresh'] + 1)
        return cache.predict(frame, deep), new, dict(full_refresh=None, reset=None, written=written)


def fixed_clip(cache, ids, stream):
    state, scans = None, []
    policy = CausalPolicy(cache, 'residual', None)
    for index, sample in enumerate(stream):
        meta = sample['meta']
        assert meta['frame'] == ids[index]
        ego = to_device(sample['ego'], torch.device('cuda'))
        spatial = spatial_input(cache.source, ego)
        refresh = cache.refresh_reason(state, meta)
        request, written, valid = None, None, None
        if refresh is None:
            frame = cache.prepare(spatial, state, meta)
            mandatory = cache.mandatory(frame)
            request = policy(frame, mandatory)
            written = mandatory if request is None else (mandatory | request)
            valid = frame.valid
        _, state, info = cache.step_spatial(spatial, state, meta, request)
        assert info['full_refresh'] == refresh
        scans.append(Scan(meta['frame'], meta, spatial, ego['label_dict'] if meta['supervised'] else None,
                          refresh, request, written, valid))
    assert len(scans) == len(ids)
    return scans


def window_start(scans):
    for end in range(WINDOW - 1, len(scans)):
        window = scans[end - WINDOW + 1:end + 1]
        if window[-1].target is None or any(scan.refresh is not None for scan in window):
            continue
        last = window[-1]
        if bool((last.valid & ~last.written).any()) and bool((~last.valid).any()):
            return end - WINDOW + 1
    return None


def select_clips(source, plan, rows, cfg, out):
    chosen = []
    cache = DeepBlockCache(source, TILE, 'light', crop_first=True)
    with torch.no_grad(), (out / 'fixed_trace.jsonl').open('x', buffering=1) as raw:
        for clip, ids, _, stream in clip_streams(plan, rows, cfg):
            scene, segment = plan['scenes'][clip]
            scans = fixed_clip(cache, ids, stream)
            assert all(scan.meta['segment'] == segment for scan in scans)
            start = window_start(scans)
            if start is None or any(item[0] == scene for item in chosen):
                continue
            for index, scan in enumerate(scans):
                raw.write(json.dumps(dict(scene=scene, clip=clip, frame=scan.frame, refresh=scan.refresh,
                    supervised=scan.target is not None, in_training_window=start <= index < start + WINDOW,
                    optional_flat_indices=None if scan.request is None else scan.request.flatten().nonzero().flatten().tolist(),
                    written_flat_indices=None if scan.written is None else scan.written.flatten().nonzero().flatten().tolist())) + '\n')
            chosen.append((scene, scans, start))
            if len(chosen) == 2:
                break
    assert len(chosen) == 2, 'Need two distinct scene clips with valid reuse and a real final label'
    return chosen


def locked_request(scan):
    if scan.refresh is not None:
        return None

    def fixed(frame, mandatory):
        written = mandatory if scan.request is None else (mandatory | scan.request)
        assert torch.equal(frame.valid, scan.valid) and torch.equal(written, scan.written), scan.frame
        return scan.request

    return fixed


def measured_parity(dense, fast, clips, raw, phase):
    records = []
    with torch.no_grad():
        for scene, scans, _ in clips:
            ref_state = local_state = None
            for scan in scans:
                request = locked_request(scan)
                (ref, ref_state, ref_info), dense_flops = measured_step(dense, scan.spatial, ref_state, scan.meta, request)
                (local, local_state, local_info), fast_flops = measured_step(fast, scan.spatial, local_state, scan.meta, request)
                assert ref_info['full_refresh'] == local_info['full_refresh'] == scan.refresh, scan.frame
                assert all(torch.equal(ref[key], local[key]) for key in ref), (phase, scan.frame, 'prediction')
                assert all(torch.equal(ref_state[key], local_state[key]) for key in STATE_TENSORS), (phase, scan.frame, 'state')
                for key in ('pose', 'scene', 'timestamp_us', 'refresh_timestamp_us', 'since_refresh'):
                    assert equal_tree(ref_state[key], local_state[key]), (phase, scan.frame, key)
                assert fast_flops <= dense_flops
                record = dict(scene=scene, frame=scan.frame, phase=phase, refresh=scan.refresh,
                    prediction_sha256=tensor_digest(ref), state_sha256=tensor_digest({key:ref_state[key] for key in STATE_TENSORS}),
                    dense_core_flops=dense_flops, local_core_flops=fast_flops)
                records.append(record)
                raw.write(json.dumps(record) + '\n')
    return records


def train_step(dense, scene_clip, criterion, optimizer):
    scene, scans, start = scene_clip
    state = None
    for scan in scans[:start]:
        with torch.no_grad():
            _, state, _ = dense.step_spatial(scan.spatial, state, scan.meta, locked_request(scan))
    state = dict(state, deep=state['deep'].detach().requires_grad_(True))
    entering_deep = state['deep']
    saved, written_valid = [], []
    optimizer.zero_grad(set_to_none=True)
    for scan in scans[start:start + WINDOW]:
        prediction, state, info = dense.step_spatial(scan.spatial, state, scan.meta, locked_request(scan))
        assert info['full_refresh'] is None
        state['deep'].retain_grad()
        dense.aligned.retain_grad()
        saved.append((state['deep'], dense.aligned))
        written_valid.append((info['written'], scan.valid))
    loss = criterion(prediction, scan.target)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert entering_deep.grad is not None and bool(torch.isfinite(entering_deep.grad).all())
    assert float(entering_deep.grad.abs().sum()) > 0
    audit = []
    for (deep, aligned), (written, valid), scan in zip(saved, written_valid, scans[start:start + WINDOW]):
        assert deep.grad is not None and aligned.grad is not None and bool(torch.isfinite(aligned.grad).all())
        assert bool(torch.isfinite(deep.grad).all()) and float(deep.grad.abs().sum()) > 0
        absolute = aligned.grad.abs()
        invalid = float(absolute.masked_select(~valid).sum())
        valid_grad = float(absolute.masked_select(valid).sum())
        reused = float(absolute.masked_select(valid & ~written).sum())
        assert invalid == 0. and valid_grad > 0.
        audit.append(dict(frame=scan.frame, deep_l1=float(deep.grad.abs().sum()), valid_aligned_l1=valid_grad,
                          invalid_aligned_l1=invalid, valid_reused_aligned_l1=reused,
                          invalid_cells=int((~valid).sum()), valid_reuse_cells=int((valid & ~written).sum())))
    assert audit[-1]['valid_reused_aligned_l1'] > 0 and audit[-1]['invalid_cells'] > 0
    grads = {}
    for name, parameter in dense.cache.source.named_parameters():
        if name not in TRAINED:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
            grads[name] = float(parameter.grad.abs().sum())
    assert tuple(grads) == TRAINED and all(value > 0 for value in grads.values())
    optimizer.step()
    return dict(scene=scene, frames=[scan.frame for scan in scans[start:start + WINDOW]], final_label=scan.frame,
        loss=float(loss.detach()), entering_deep_l1=float(entering_deep.grad.abs().sum()), grad_l1=grads, history_gradients=audit,
        prediction_sha256=tensor_digest(prediction), state_sha256=tensor_digest({key:state[key] for key in STATE_TENSORS}))


def make_trainable(source):
    model = copy.deepcopy(source)
    assert model_digest(model) == model_digest(source)
    named = dict(model.named_parameters())
    for key in TRAINED:
        named[key].requires_grad_(True)
    assert tuple(key for key, value in named.items() if value.requires_grad) == TRAINED
    return model, torch.optim.AdamW([named[key] for key in TRAINED], lr=LR, weight_decay=WEIGHT_DECAY)


def save_gate(path, source, optimizer, identity, next_window):
    weights = {name: source.state_dict()[name].detach().cpu() for name in TRAINED}
    atomic_save(dict(weights=weights, optimizer=optimizer.state_dict(), rng=capture_rng(), identity=identity,
                     next_window=next_window), path)


def load_gate(path, source, optimizer, identity, next_window):
    saved = torch.load(path, map_location='cpu')
    assert saved['identity'] == identity and saved['next_window'] == next_window, path
    assert set(saved['weights']) == set(TRAINED)
    named = dict(source.named_parameters())
    with torch.no_grad():
        for name, value in saved['weights'].items():
            named[name].copy_(value)
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    assert equal_tree(capture_rng(), saved['rng'])


def run(args, out):
    plan_hash = digest(Path(args.plan))
    plan = checked_plan(args.plan)
    assert plan['split'] == 'train' and len(plan['clips']) == len(plan['scenes'])
    args.kind = 'quality'
    source, cfg, rows, provenance = build(args)
    assert provenance['split'] == plan['split']
    source.period, source.max_age = 5, .55
    random.seed(SEED)
    parent_digest = model_digest(source)
    frozen_digest = tensor_digest(source.state_dict(), TRAINED)
    model, optimizer = make_trainable(source)
    dense, fast = DenseOrdinaryReference(model), DeepBlockCache(model, TILE, 'light', crop_first=True)
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    clips = select_clips(source, plan, rows, cfg, out)
    trained_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    identity = dict(plan_sha256=plan_hash, source_sha256=SOURCE_SHA, frozen_digest=frozen_digest,
        trace_sha256=digest(out / 'fixed_trace.jsonl'), windows=[[scan.frame for scan in scans[start:start + WINDOW]]
                     for _, scans, start in clips], trained=TRAINED, lr=LR, weight_decay=WEIGHT_DECAY)
    write_json(out / 'contract.json', dict(identity, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        device=torch.cuda.get_device_name(), torch=torch.__version__, period=5, max_age=.55, tile=TILE, cap=CAP,
        training='Two fixed three-ordinary-scan windows, one final-label AdamW step each; second repeated after restore.',
        scope='Frozen P5 geometric residual masks. Graph only through ordinary deep cache and frozen readout. Full refresh and warmup prefix under no_grad.',
        timing_ap='Engineering loss and actual Conv/ConvTranspose counts; no validation AP or timing.'))
    with (out / 'frames.jsonl').open('x', buffering=1) as raw:
        initial = measured_parity(dense, fast, clips, raw, 'before')
        initial_train = tensor_digest({key:model.state_dict()[key] for key in TRAINED})
        write_json(out / 'status.json', dict(status='RUNNING', phase='gradient_and_first_step'))
        first = train_step(dense, clips[0], criterion, optimizer)
        assert any(r['frame'] == first['final_label'] and r['prediction_sha256'] == first['prediction_sha256']
                   and r['state_sha256'] == first['state_sha256'] for r in initial), 'Gradient-mode forward differs'
        assert tensor_digest({key:model.state_dict()[key] for key in TRAINED}) != initial_train
        assert tensor_digest(model.state_dict(), TRAINED) == frozen_digest
        checkpoint = out / 'after_step1.pth'
        save_gate(checkpoint, model, optimizer, identity, 1)
        updated = measured_parity(dense, fast, clips, raw, 'after_first_step')
        write_json(out / 'status.json', dict(status='RUNNING', phase='exact_resume'))
        second = train_step(dense, clips[1], criterion, optimizer)
        baseline_rng = capture_rng()
        resumed, resumed_optimizer = make_trainable(source)
        load_gate(checkpoint, resumed, resumed_optimizer, identity, 1)
        repeated = train_step(DenseOrdinaryReference(resumed), clips[1],
                              PointPillarDirLoss(cfg['loss']['args']).cuda().eval(), resumed_optimizer)
        assert repeated == second, 'Second-window training replay differs'
        assert equal_tree(optimizer.state_dict(), resumed_optimizer.state_dict()), 'Resumed optimizer differs'
        assert equal_tree(capture_rng(), baseline_rng), 'Resumed RNG differs'
        assert model_digest(model) == model_digest(resumed), 'Resumed weights differ'
    assert tensor_digest(resumed.state_dict(), TRAINED) == tensor_digest(model.state_dict(), TRAINED) == frozen_digest
    assert model_digest(source) == parent_digest and digest(Path(args.plan)) == plan_hash
    write_json(out / 'report.json', dict(status='COMPLETED', parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=trained_size, fixed_trace_scans=len(initial), parity_scans_before=len(initial),
        parity_scans_after=len(updated), fixed_masks_exact=True, predictions_and_states_exact=True,
        frozen_weights_and_buffers_exact=True, source_checkpoint_unchanged=True,
        checkpoint_sha256=digest(checkpoint), atomic_resume_exact=True, resume_weights_sha256=model_digest(resumed),
        dense_core_flops_mean=mean(r['dense_core_flops'] for r in updated),
        local_core_flops_mean=mean(r['local_core_flops'] for r in updated),
        first=first, second=second, resumed_second=repeated, ap50=None, ap70=None, model_step_ms_mean=None))
    write_json(out / 'status.json', dict(status='COMPLETED', compared_scans=len(initial), verified_train_windows=2))


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

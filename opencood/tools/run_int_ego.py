"""Bounded INT FM-only train/eval on existing DAIR ego sequences.

This is an OpenCOOD adaptation, not official Waymo/nuScenes benchmark reproduction.
Every run is isolated and all checkpoints have explicit names, never mutable best.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from opencood.data_utils.int_ego_stream import (
    DairEgoStreamDataset, build_stream_manifest, training_order, stream_collate, sha256)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.point_pillar_int import PointPillarINT
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def loader(rows, config, workers, seed):
    dataset = DairEgoStreamDataset(rows, config, seed)
    stream = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=stream_collate, generator=torch.Generator().manual_seed(seed))
    return dataset, stream


def train(model, rows, config, args, directory, optimizer=None, order=None,
          full_epoch=False, save_optimizer=True):
    if order is None:
        order = training_order(rows, args.clip_length, args.seed)
    _, stream = loader(order, config, args.workers, args.seed)
    model.train()
    freeze_bn_stats(model)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = PointPillarDirLoss(config['loss']['args']).cuda()
    state, updates, scans = None, 0, 0
    resets, losses = Counter(), []
    started = time.monotonic()
    with (directory/'train_frames.jsonl').open('w', buffering=1) as log:
        for sample in stream:
            if not full_epoch and updates >= args.train_steps:
                break
            meta = sample['meta']
            if sample['ego'] is None:
                state = None
                log.write(json.dumps(dict(frame=meta['frame'], invalid=meta['invalid']))+'\n')
                continue
            ego = to_device(sample['ego'], torch.device('cuda'))
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(meta['supervised']):
                pred, state, info = model.step(ego, state, meta, reset=meta['reset_before'])
                entry = dict(frame=meta['frame'], scene=meta['segment'], supervised=meta['supervised'], **info)
                if meta['supervised']:
                    loss = criterion(pred, ego['label_dict'])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss at '+meta['frame'])
                    loss.backward()
                    temporal_grad = sum(float(p.grad.detach().square().sum()) for p in model.feature_memory.parameters() if p.grad is not None)**.5
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
                    if not np.isfinite(norm):
                        raise RuntimeError('Non-finite gradient')
                    optimizer.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                    entry.update(update=updates, loss=losses[-1], gradient_norm=norm, temporal_gradient_norm=temporal_grad)
            scans += 1
            if info['reset']:
                resets[info['reset']] += 1
            log.write(json.dumps(entry)+'\n')
            if updates and updates % 32 == 0 and meta['supervised']:
                print('TRAIN', args.mode, updates, 'scans', scans, 'loss', round(losses[-1], 5), flush=True)
    expected = sum(r['supervised'] for r in order) if full_epoch else args.train_steps
    if updates != expected or (full_epoch and scans != len(order)):
        raise RuntimeError('Training did not consume the required valid frames')
    checkpoint = directory/'trained.pth'
    saved = {'model': model.state_dict(), 'updates': updates,
             'mode': args.mode, 'seed': args.seed}
    if save_optimizer:
        saved['optimizer'] = optimizer.state_dict()
    torch.save(saved, str(checkpoint))
    summary = dict(updates=updates, scans=scans, resets=dict(resets),
                   loss_mean=float(np.mean(losses)),
                   first32_loss_mean=float(np.mean(losses[:32])), last32_loss_mean=float(np.mean(losses[-32:])),
                   seconds=time.monotonic()-started, checkpoint_sha256=sha256(checkpoint),
                   protocol='Per-frame detach, explicit segment/clip resets, all weights trainable; BN running statistics frozen; no augmentation')
    write_json(directory/'train_summary.json', summary)
    return summary


def evaluate(model, rows, config, args, directory):
    dataset, stream = loader(rows, config, args.workers, args.seed)
    model.eval()
    stats = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in [.3, .5, .7]}
    state, scans, labels, diagnostic_count = None, 0, 0, 0
    resets, step_ms, diagnostics = Counter(), [], []
    started = time.monotonic()
    with (directory/'eval_frames.jsonl').open('w', buffering=1) as log, torch.no_grad():
        for sample in stream:
            meta = sample['meta']
            if sample['ego'] is None:
                state = None
                log.write(json.dumps(dict(frame=meta['frame'], invalid=meta['invalid']))+'\n')
                continue
            ego = to_device(sample['ego'], torch.device('cuda'))
            previous = state
            torch.cuda.synchronize()
            tick = time.perf_counter()
            prediction, state, info = model.step(
                ego, previous, meta, reset=args.history_policy == 'reset',
                align=args.history_policy != 'no-align')
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-tick)*1000
            if any(not torch.isfinite(value).all() for value in prediction.values()):
                raise RuntimeError('Non-finite prediction at '+meta['frame'])
            if scans >= 10:
                step_ms.append(elapsed)
            entry = dict(frame=meta['frame'], scene=meta['segment'], timestamp_us=meta['timestamp_us'],
                         supervised=meta['supervised'], model_step_ms=elapsed, **info)
            scans += 1
            if info['reset']:
                resets[info['reset']] += 1
            if meta['supervised']:
                data = {'ego': ego}
                boxes, scores = dataset.post.post_process(data, {'ego': prediction})
                gt = dataset.post.generate_gt_bbx(data)
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
                labels += 1
                entry.update(gt=int(gt.shape[0]), predictions=0 if boxes is None else int(boxes.shape[0]))
                if args.history_policy == 'aligned' and args.mode != 'single' and previous is not None and info['reset'] is None and diagnostic_count < 16:
                    reset_pred, _, _ = model.step(ego, previous, meta, reset=True)
                    noalign_pred, _, _ = model.step(ego, previous, meta, align=False)
                    diag = dict(frame=meta['frame'],
                                reset_psm_l1=float((prediction['psm']-reset_pred['psm']).abs().mean()),
                                reset_rm_l1=float((prediction['rm']-reset_pred['rm']).abs().mean()),
                                noalign_psm_l1=float((prediction['psm']-noalign_pred['psm']).abs().mean()))
                    diagnostics.append(diag)
                    diagnostic_count += 1
            log.write(json.dumps(entry)+'\n')
            if scans % 400 == 0:
                print('EVAL', args.mode, 'scans', scans, '/', len(rows), 'labels', labels, flush=True)
    if labels == 0 or stats[.5]['gt'] == 0:
        raise RuntimeError('Evaluation produced no supervised data')
    metrics = {'bev_ap%d' % int(t*100): eval_utils.calculate_ap(stats, t)[0] for t in stats}
    result = dict(metrics=metrics, history_policy=args.history_policy,
                  scans=scans, labels=labels, gt_boxes=stats[.5]['gt'],
                  predicted_boxes=len(stats[.5]['score']), resets=dict(resets),
                  state_bytes=0 if state is None else state['value'].numel()*state['value'].element_size(),
                  seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  model_step_ms_p50=float(np.percentile(step_ms, 50)), model_step_ms_p95=float(np.percentile(step_ms, 95)),
                  timing_scope='Model only: pillar encoder + warp/FM + backbone + head, excludes IO/voxelization/postprocess; diagnostic replays excluded',
                  history_diagnostics=diagnostics)
    write_json(directory/'eval_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/repro/dair_stage1_codyntrust_single_wide.yaml')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', choices=['single','concat','gru'], required=True)
    parser.add_argument('--train-steps', type=int, default=256)
    parser.add_argument('--clip-length', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--history-policy', choices=['aligned', 'reset', 'no-align'],
                        default='aligned', help='Frozen-checkpoint evaluation ablation')
    args = parser.parse_args()
    if args.history_policy != 'aligned' and (not args.eval_only or args.mode == 'single'):
        parser.error('History ablations require --eval-only and a temporal model')
    # TF32 convolution can amplify identity-path rounding through the detector.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    config = load_yaml(args.config)
    config['data_augment'] = []
    config['int_mode'] = args.mode
    with (directory/'config.yaml').open('w') as f:
        yaml.dump(config, f)
    manifest = json.loads(Path(args.manifest).read_text())
    model = PointPillarINT(config['model']['args'], args.mode).cuda()
    state = torch.load(args.checkpoint, map_location='cpu')
    if args.eval_only and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
        loaded = len(state['model'])
    else:
        loaded = model.load_single_checkpoint(state)
    provenance = dict(arguments=vars(args), code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                      python=sys.executable, torch=torch.__version__, checkpoint_sha256=sha256(args.checkpoint),
                      manifest_sha256=sha256(args.manifest), strict_spatial_state_entries=loaded,
                      upstream_int='988157ff131a0c027472bd0f00c0bda0e08cded0',
                      tf32=False,
                      protocol='FM-only; PC/PM disabled; dataset reference ego poses; BEV SE(2) nearest warp; ego labels only')
    write_json(directory/'manifest.json', provenance)
    # Confirm single and identity-initialized Concat preserve pretrained detector output.
    if not args.eval_only and args.mode in ['single','concat']:
        audit_set = DairEgoStreamDataset(manifest['train'][:1], config, args.seed)
        sample = audit_set[0]
        ego = to_device(sample['ego'], torch.device('cuda'))
        reference = PointPillarCodyntrustSingle(config['model']['args']).cuda().eval()
        reference.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            expected = reference(ego)
            actual, _, _ = model.step(ego, None, sample['meta'])
        differences = {k: float((expected[k]-actual[k]).abs().max()) for k in expected}
        if max(differences.values()) > 1e-3:
            raise RuntimeError('P1 identity compatibility failed: %s' % differences)
        write_json(directory/'identity_audit.json', differences)
        del reference, expected, actual, ego, sample
        torch.cuda.empty_cache()
    del state
    training = None if args.eval_only else train(model, manifest['train'], config, args, directory)
    # Reload an explicit frozen checkpoint into a fresh model before evaluation.
    if training is not None:
        del model
        torch.cuda.empty_cache()
        model = PointPillarINT(config['model']['args'], args.mode).cuda()
        model.load_state_dict(torch.load(str(directory/'trained.pth'), map_location='cpu')['model'], strict=True)
    torch.cuda.reset_peak_memory_stats()
    evaluation = evaluate(model, manifest['val'], config, args, directory)
    write_json(directory/'result.json', dict(status='COMPLETED', training=training, evaluation=evaluation))
    print('COMPLETED', args.mode, json.dumps(evaluation['metrics']), flush=True)


if __name__ == '__main__':
    main()

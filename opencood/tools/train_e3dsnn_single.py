"""Reproducible vehicle-only E-3DSNN adaptation: overfit, train, evaluate.

Writes bounded checkpoints (last resume state + best bare state_dict), explicit
sample counts and full-validation BEV AP. Never silently skips bad samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_torch(path, value):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, str(temporary))
    temporary.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def checked_forward(model, criterion, batch):
    output = model(batch['ego'])
    labels = batch['ego']['label_dict']
    assert output['psm'].shape[2:] == labels['pos_equal_one'].shape[1:3], (
        'Prediction/anchor grid mismatch', output['psm'].shape, labels['pos_equal_one'].shape)
    assert all(torch.isfinite(output[key]).all() for key in ('psm', 'rm', 'dm'))
    loss = criterion(output, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite detection loss')
    return output, loss


def validate(model, dataset, workers, limit=0):
    model.eval()
    stat = {threshold: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for threshold in (.3, .5, .7)}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=dataset.collate_batch_test, worker_init_fn=seed_worker,
                        **({'multiprocessing_context': 'spawn'} if workers else {}))
    evaluated, predicted, ground_truth = 0, 0, 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            if batch is None or set(batch) != {'ego'}:
                raise ValueError('Invalid or non-single-agent validation batch at {}'.format(index))
            batch = train_utils.to_device(batch, torch.device('cuda'))
            output = model(batch['ego'])
            boxes, scores, gt = dataset.post_process_no_fusion(batch, {'ego': output})
            if boxes is not None and not torch.isfinite(boxes).all():
                raise FloatingPointError('Nonfinite decoded boxes')
            for threshold in stat:
                eval_utils.caluclate_tp_fp(boxes, scores, gt, stat, threshold)
            evaluated += 1
            predicted += 0 if boxes is None else len(boxes)
            ground_truth += len(gt)
    return {'samples': evaluated, 'indexed_samples': len(dataset),
            'full_split': evaluated == len(dataset), 'predicted_boxes': predicted,
            'gt_boxes': ground_truth, 'metric': 'BEV AP, vehicle-local GT, no fusion',
            'ap': {str(threshold): eval_utils.calculate_ap(stat, threshold)[0] for threshold in stat}}


def execute(args, output_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    cfg = load_yaml(args.config)
    if cfg['fusion']['core_method'] != 'SingleDAIRVehicle':
        raise ValueError('This runner requires the explicit vehicle-only dataset')
    if args.mode == 'overfit':
        cfg['data_augment'] = []
    train_data = build_dataset(cfg, train=True)
    val_data = build_dataset(cfg, train=False)
    model = train_utils.create_model(cfg).cuda()
    criterion = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=cfg['lr_scheduler']['step_size'], gamma=cfg['lr_scheduler']['gamma'])
    generator = torch.Generator().manual_seed(args.seed)
    manifest = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'arguments': vars(args), 'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'split_sha256': {key: hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
                         for key in ('root_dir', 'validate_dir')},
        'data_manifest_sha256': hashlib.sha256(Path(cfg['data_manifest']).read_bytes()).hexdigest(),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'gpu_visible': os.getenv('CUDA_VISIBLE_DEVICES'), 'parameters': sum(p.numel() for p in model.parameters()),
        'architecture': cfg['model']['core_method'],
        'activation': cfg['model']['args'].get('activation', 'count4'),
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
        'train_samples': len(train_data), 'val_samples': len(val_data),
        'protocol': 'vehicle-only point cloud and local GT; no history/infra; see architecture for model'}
    write_json(output_dir / ('manifest_{}.json'.format(int(time.time()))), manifest)
    (output_dir / 'config.yaml').write_bytes(Path(args.config).read_bytes())
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    if args.mode == 'eval':
        if not args.checkpoint:
            raise ValueError('Evaluation requires --checkpoint')
        metrics = validate(model, val_data, args.workers)
        write_json(output_dir / 'evaluation.json', metrics)
        return metrics
    if args.mode == 'overfit':
        batch = train_data.collate_batch_train([train_data[args.sample_index]])
        batch = train_utils.to_device(batch, torch.device('cuda'))
        batch['ego']['return_spikes'] = True
        losses, gradients, hist = [], {}, None
        model.train()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            result, loss = checked_forward(model, criterion, batch)
            loss.backward()
            if step == 0:
                for name in ('backbone_3d.conv_input.0.weight', 'backbone_3d.conv_out.0.weight',
                             'backbone_2d.blocks.0.2.weight', 'reg_head.weight'):
                    grad = dict(model.named_parameters())[name].grad
                    if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                        raise ValueError('Missing/nonfinite/zero gradient: ' + name)
                    gradients[name] = float(grad.abs().sum())
                spikes = result['spike_features'].detach()
                if cfg['model']['args'].get('activation', 'count4') == 'count4':
                    assert torch.equal(spikes, spikes.round()) and spikes.min() >= 0 and spikes.max() <= 4
                    hist = torch.bincount(spikes.long().flatten(), minlength=5).cpu().tolist()
                else:
                    assert torch.isfinite(spikes).all() and spikes.min() >= 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 10 == 0 or step == args.steps - 1:
                print(json.dumps({'step': step + 1, 'loss': losses[-1]}), flush=True)
                write_json(output_dir / 'progress.json', {'mode': 'overfit', 'step': step + 1, 'loss': losses[-1]})
        # A reduction is a wiring gate, not proof of convergence/generalization.
        passed = float(np.mean(losses[-10:])) < float(np.mean(losses[:10])) * .5
        result = {'passed': passed, 'losses': losses, 'gradients': gradients,
            'initial_count_histogram': hist, 'activation': cfg['model']['args'].get('activation', 'count4'),
            'vehicle_frame': train_data.data[args.sample_index],
            'positive_anchors': int(batch['ego']['label_dict']['pos_equal_one'].sum()),
            'gate': 'last-10 mean loss < 50% of first-10 mean loss',
            'decode_check': validate(model, val_data, 0, limit=2),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        save_torch(output_dir / 'overfit.pth', model.state_dict())
        write_json(output_dir / 'overfit.json', result)
        if not passed:
            raise RuntimeError('Single-batch loss reduction gate failed; inspect overfit.json')
        return result
    start_epoch, best = 0, -1.
    if args.resume:
        state = torch.load(str(output_dir / 'last.pt'), map_location='cpu')
        if state['config_sha256'] != manifest['config_sha256'] or state['data_manifest_sha256'] != manifest['data_manifest_sha256']:
            raise ValueError('Resume requires the same config and data manifest')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best = state['epoch'], state['best_ap50']
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        torch.cuda.set_rng_state_all(state['cuda_state'])
        generator.set_state(state['loader_state'])
    loader = DataLoader(train_data, batch_size=cfg['train_params']['batch_size'], shuffle=True,
        num_workers=args.workers, collate_fn=train_data.collate_batch_train, pin_memory=True,
        drop_last=False, worker_init_fn=seed_worker, generator=generator,
        **({'multiprocessing_context': 'spawn'} if args.workers else {}))
    epochs = args.epochs or cfg['train_params']['epoches']
    for epoch in range(start_epoch, epochs):
        model.train()
        total, samples = 0., 0
        started = time.monotonic()
        for index, batch in enumerate(loader):
            if batch is None:
                raise ValueError('Unexpected empty training batch')
            batch = train_utils.to_device(batch, torch.device('cuda'))
            optimizer.zero_grad(set_to_none=True)
            _, loss = checked_forward(model, criterion, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            n = batch['ego']['object_bbx_mask'].shape[0]
            total += float(loss.detach()) * n
            samples += n
            if index % 50 == 0:
                progress = {'state': 'training', 'epoch': epoch + 1, 'epochs': epochs,
                    'batch': index + 1, 'batches': len(loader), 'loss': float(loss.detach()),
                    'gradient_norm': float(grad_norm), 'elapsed_seconds': time.monotonic() - started}
                write_json(output_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        assert samples == len(train_data)
        scheduler.step()
        write_json(output_dir / 'progress.json', {'state': 'validating', 'epoch': epoch + 1})
        metrics = validate(model, val_data, args.workers)
        metrics.update({'epoch': epoch + 1, 'train_loss': total / samples,
                        'train_samples': samples, 'seconds': time.monotonic() - started})
        if metrics['ap']['0.5'] > best:
            best = metrics['ap']['0.5']
            save_torch(output_dir / 'best.pth', model.state_dict())
            write_json(output_dir / 'best_metrics.json', metrics)
        save_torch(output_dir / 'last.pt', {'epoch': epoch + 1, 'model': model.state_dict(),
            'config_sha256': manifest['config_sha256'], 'data_manifest_sha256': manifest['data_manifest_sha256'],
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_ap50': best,
            'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(), 'cuda_state': torch.cuda.get_rng_state_all(),
            'loader_state': generator.get_state()})
        with (output_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    return {'epochs_completed': epochs, 'best_ap50': best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['overfit', 'train', 'eval'], required=True)
    parser.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh output directory or explicit --resume')
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(args, output_dir)
        write_json(output_dir / 'status.json', {'state': 'complete', 'mode': args.mode,
                                              'finished_unix': time.time()})
        print('Completed ' + args.mode, flush=True)
    except Exception:
        write_json(output_dir / 'status.json', {'state': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

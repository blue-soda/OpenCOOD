"""CAPRI DAIR training/evaluation. No ECTRA model or flow compensation is used."""
import argparse
import copy
import distutils.version
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.getcwd())
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.capri.sender import CapriSender
from opencood.models.capri.model import Capri, capri_loss
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, eval_utils


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def packet_bytes(message):
    return sum(v.numel()*v.element_size() for v in message.values() if torch.is_tensor(v)) + sum(
        v.numel()*v.element_size() for v in message['background'].values()) + 16*4 + 8


def prepare(batch, sender, dataset, cfg):
    data = batch['ego']
    assert len(data['record_len']) == 1 and int(data['record_len'][0]) == 2, 'v1 supports DAIR batch=1, two agents'
    k = int(data['past_k_time_interval'].numel()//2)
    messages = sender.messages(data['processed_lidar'], 2*k, data['anchor_box'], dataset.post_processor)
    history = data['ectra_ego_history']
    refs = sender.messages(history['processed_lidar'], k, data['anchor_box'], dataset.post_processor)
    times = data['past_k_time_interval'].reshape(2, k)*cfg['frame_period_s']
    arguments = (messages, data['pairwise_t_matrix'][0, :2, :k], times,
                 refs, history['to_current_ego'][0], history['valid'][0])
    communication = sum(packet_bytes(m) for m in messages[k:])
    return arguments, communication


def predictions(output, postprocessor):
    scores = output['logits'].sigmoid().detach()
    boxes = output['boxes'].detach()
    keep = scores > postprocessor.params['target_args']['score_threshold']
    corners = box_utils.boxes_to_corners_3d(boxes[keep], order='hwl')
    scores = scores[keep]
    if len(corners):
        keep = box_utils.remove_large_pred_bbx(corners) & box_utils.remove_bbx_abnormal_z(corners)
        corners, scores = corners[keep], scores[keep]
        keep = box_utils.nms_rotated(corners, scores, postprocessor.params['nms_thresh'])
        corners, scores = corners[keep], scores[keep]
        keep = box_utils.get_mask_for_boxes_within_range_torch(corners, postprocessor.params['gt_range'])
        corners, scores = corners[keep], scores[keep]
    return corners, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--max_steps', type=int, default=0, help='Smoke limit; 0 means full split')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode == 'eval' and not args.checkpoint:
        parser.error('Evaluation requires an explicit CAPRI checkpoint')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    base = load_yaml(cfg['base_config'])
    base['ectra_ego_history'] = {'enabled': True, 'max_age_ms': 100}
    base['train_params']['batch_size'] = 1
    (out/'data_config.yaml').write_text(yaml.dump(base))
    seed(cfg['seed'] if args.mode == 'train' else cfg['eval_seed'])
    dataset = build_dataset(copy.deepcopy(base), visualize=False, train=args.mode == 'train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sender = CapriSender(base['model']['args'], cfg['sender']).to(device)
    sender.load_state_dict(torch.load(cfg['single_checkpoint'], map_location='cpu'), strict=True)
    sender.eval()
    model = Capri(sender.out_channel, cfg['model']).to(device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu')
        if saved['config'] != cfg:
            raise ValueError('Checkpoint configuration differs from requested configuration')
        model.load_state_dict(saved['model'], strict=True)
    model.train(args.mode == 'train')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    collate = dataset.collate_batch_train if args.mode == 'train' else dataset.collate_batch_test
    loader = DataLoader(dataset, batch_size=1, shuffle=args.mode == 'train', num_workers=args.workers, collate_fn=collate)
    metadata = {'arguments': vars(args), 'single_sha256': digest(cfg['single_checkpoint']),
                'checkpoint_sha256': digest(args.checkpoint) if args.checkpoint else None,
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'parameters': sum(p.numel() for p in model.parameters()),
                'modules': {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
                'smoke': args.max_steps > 0, 'started_at': time.time()}
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2))
    writer = SummaryWriter(str(out/'tensorboard'))
    epochs = (args.epochs or cfg['epochs']) if args.mode == 'train' else 1
    total_step = 0
    for epoch in range(epochs):
        statistics = {x: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for x in (.3, .5, .7)}
        identities, losses = [], []
        for index, batch in enumerate(loader):
            if args.max_steps and index >= args.max_steps:
                break
            if batch is None:
                continue
            batch = to_device(batch, device)
            arguments, communication = prepare(batch, sender, dataset, cfg)
            data = batch['ego']
            with torch.set_grad_enabled(args.mode == 'train'):
                result = model(*arguments)
                gt = data['object_bbx_center'][0][data['object_bbx_mask'][0].bool()]
                loss, parts = capri_loss(result, gt)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite CAPRI loss')
            gradients = {}
            if args.mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                for name, module in model.named_children():
                    gradients[name] = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                if not all(np.isfinite(v) for v in gradients.values()):
                    raise FloatingPointError('Non-finite CAPRI gradient')
                assert all(p.grad is None for p in sender.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
            else:
                corners, scores = predictions(result, dataset.post_processor)
                gt_corners = dataset.post_processor.generate_gt_bbx(batch)
                for threshold in statistics:
                    eval_utils.caluclate_tp_fp(corners, scores, gt_corners, statistics, threshold)
            sample = data.get('sample_idx', index)
            if torch.is_tensor(sample):
                sample = sample.detach().cpu().tolist()
            if isinstance(sample, (list, tuple)) and len(sample) == 1:
                sample = sample[0]
            intervals = data['past_k_time_interval'].detach().cpu().reshape(-1).tolist()
            identities.append((sample, intervals))
            row = dict(epoch=epoch+1, step=index, loss=float(loss.detach()), sample_idx=sample,
                       time_intervals=intervals, communication_bytes=communication, **parts,
                       **result['diagnostics'])
            if args.mode == 'train' and (index < 3 or index % 100 == 0):
                row['gradients'] = gradients
            with (out/'scalars.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, total_step)
            total_step += 1
            losses.append(row['loss'])
            if index < 3 or index % 20 == 0:
                print(json.dumps(row), flush=True)
        if not losses:
            raise RuntimeError('No usable samples')
        summary = {'epoch': epoch+1, 'mean_loss': sum(losses)/len(losses), 'samples': len(losses),
                   'sample_order_sha256': hashlib.sha256(json.dumps(identities).encode()).hexdigest()}
        if args.mode == 'train':
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': cfg, 'epoch': epoch+1}, str(out/('capri_epoch%d.pth' % (epoch+1))))
        else:
            summary['ap30'], summary['ap50'], summary['ap70'] = eval_utils.eval_final_results(statistics, str(out), dataset='d')
        (out/('summary_epoch%d.json' % (epoch+1))).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    writer.close()


if __name__ == '__main__':
    main()

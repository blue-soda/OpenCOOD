# -*- coding: utf-8 -*-
"""Diagnose single-branch score consistency for Stage2 ROI generation."""

import argparse
import copy
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2_yaml', required=True)
    parser.add_argument('--single_yaml', required=True)
    parser.add_argument('--checkpoint', required=True,
                        help='single checkpoint .pth or directory')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--thresholds', default='0.1,0.2,0.3,0.5')
    return parser.parse_args()


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    best = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.startswith('net_epoch_bestval_at') and name.endswith('.pth')
    ]
    if best:
        return best[0]
    epochs = []
    for name in os.listdir(path):
        if name.startswith('net_epoch') and name.endswith('.pth'):
            try:
                epoch = int(name[len('net_epoch'):-len('.pth')])
            except ValueError:
                continue
            epochs.append((epoch, os.path.join(path, name)))
    if not epochs:
        raise FileNotFoundError('no checkpoint found in %s' % path)
    return sorted(epochs)[-1][1]


def filtered_load_state_dict(model, checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')
    model_state = model.state_dict()
    usable = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            usable[key] = value
        else:
            skipped.append(key)
    missing = [key for key in model_state if key not in usable]
    model.load_state_dict(usable, strict=False)
    return len(usable), skipped, missing


def with_lidar(data_dict, processed_lidar):
    cloned = dict(data_dict)
    cloned['processed_lidar'] = processed_lidar
    return cloned


def score_stats(name, psm, thresholds):
    prob = torch.sigmoid(psm.detach()).reshape(psm.shape[0], -1)
    max_score = prob.max(dim=1)[0]
    mean_score = prob.mean(dim=1)
    parts = [
        '%s: shape=%s' % (name, tuple(psm.shape)),
        'max_mean=%.5f max_min=%.5f max_max=%.5f' % (
            max_score.mean().item(), max_score.min().item(),
            max_score.max().item()),
        'mean_prob=%.5f' % mean_score.mean().item(),
    ]
    for threshold in thresholds:
        counts = (prob > threshold).sum(dim=1).float()
        parts.append('cnt>%.2f_mean=%.2f max=%.0f' % (
            threshold, counts.mean().item(), counts.max().item()))
    return ' | '.join(parts)


def box_count(dataset, psm, rm, dm, batch, thresholds):
    old_threshold = dataset.post_processor.params['target_args']['score_threshold']
    counts = {}
    anchor_box = batch['anchor_box']
    k = int(batch['past_k_sample_interval'].numel() //
            max(int(batch['record_len'].sum().item()), 1))
    if k <= 0:
        k = 1
    trans = torch.eye(4, device=psm.device).reshape(1, 4, 4).repeat(
        psm.shape[0], 1, 1)
    past_time = torch.zeros(psm.shape[0], device=psm.device)
    m_single = {'psm_single': psm, 'rm_single': rm}
    if dm is not None:
        m_single['dm_single'] = dm
    try:
        for threshold in thresholds:
            dataset.post_processor.params['target_args']['score_threshold'] = threshold
            result = dataset.generate_pred_bbx_frames(
                m_single, trans, past_time, anchor_box)
            frame_counts = []
            for _, frame_result in result.items():
                box_tensor = frame_result.get('pred_box_center_tensor', None)
                frame_counts.append(0 if box_tensor is None else int(box_tensor.shape[0]))
            counts[threshold] = frame_counts
    finally:
        dataset.post_processor.params['target_args']['score_threshold'] = old_threshold
    return counts


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip()]
    checkpoint = resolve_checkpoint(args.checkpoint)

    stage2_hypes = yaml_utils.load_yaml(args.stage2_yaml)
    single_hypes = yaml_utils.load_yaml(args.single_yaml)

    dataset = build_dataset(stage2_hypes, visualize=False, train=False)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False)

    single_model = train_utils.create_model(single_hypes)
    used, skipped, missing = filtered_load_state_dict(single_model, checkpoint)
    print('[single-load] checkpoint=%s' % checkpoint)
    print('[single-load] loaded=%d skipped=%d missing=%d' %
          (used, len(skipped), len(missing)))
    if skipped[:5]:
        print('[single-load] skipped_head=%s' % skipped[:5])
    if missing[:5]:
        print('[single-load] missing_head=%s' % missing[:5])

    stage2_model = train_utils.create_model(stage2_hypes)
    used2, skipped2, missing2 = filtered_load_state_dict(stage2_model, checkpoint)
    print('[stage2-single-load] loaded=%d skipped=%d missing=%d' %
          (used2, len(skipped2), len(missing2)))
    if skipped2[:5]:
        print('[stage2-single-load] skipped_head=%s' % skipped2[:5])
    if missing2[:5]:
        print('[stage2-single-load] missing_head=%s' % missing2[:5])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    single_model = single_model.to(device).eval()
    stage2_model = stage2_model.to(device).eval()

    for sample_id, batch_data in enumerate(loader):
        if sample_id >= args.limit:
            break
        batch = train_utils.to_device(batch_data['ego'], device)
        sample_idx = batch.get('sample_idx', sample_id)
        if torch.is_tensor(sample_idx):
            sample_idx = sample_idx.flatten()[0].item()

        with torch.no_grad():
            current_output = single_model(
                with_lidar(batch, batch['curr_processed_lidar']))
            history_output = single_model(
                with_lidar(batch, batch['processed_lidar']))
            stage2_output = stage2_model(batch, dataset)

        print('[sample %s]' % sample_idx)
        print('  ' + score_stats('single/current',
                                 current_output['psm'], thresholds))
        print('  ' + score_stats('single/history',
                                 history_output['psm'], thresholds))
        print('  ' + score_stats('stage2/internal',
                                 stage2_output['psm_single'], thresholds))

        try:
            current_counts = box_count(
                dataset, current_output['psm'], current_output['rm'],
                current_output.get('dm'), batch, thresholds)
            history_counts = box_count(
                dataset, history_output['psm'], history_output['rm'],
                history_output.get('dm'), batch, thresholds)
            print('  decoded/current=%s' % current_counts)
            print('  decoded/history=%s' % history_counts)
        except Exception as exc:
            print('  decoded/error=%s: %s' % (type(exc).__name__, exc))


if __name__ == '__main__':
    main()

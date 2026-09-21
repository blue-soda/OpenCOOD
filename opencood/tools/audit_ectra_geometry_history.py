"""Finite real-data parity, causal reference and gradient audit before training."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.train import add_auxiliary_losses


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1,
                        help='space audit indices across scenes instead of adjacent frames')
    parser.add_argument('--input_gradients', action='store_true',
                        help='probe detection/total loss sensitivity to recurrent input branches')
    args = parser.parse_args()
    if args.samples < 1 or args.start < 0 or args.stride < 1:
        parser.error('samples/stride must be positive and start nonnegative')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(str(destination))
    cfg = load_yaml(args.config)
    original_cfg = copy.deepcopy(cfg)
    original_cfg['ectra_ego_history'] = {'enabled': False}
    baseline = build_dataset(original_cfg, visualize=False, train=False)
    dataset = build_dataset(cfg, visualize=False, train=False)
    model = train_utils.create_model(cfg).cuda().train()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    criterion = train_utils.create_loss(cfg)
    probe = None
    if args.input_gradients:
        from opencood.tools.ectra_gradient_diagnostics import EctraInputGradientProbe
        probe = EctraInputGradientProbe(model.ectra)
    rows = []
    if args.start >= len(dataset):
        parser.error('start is outside the dataset')
    for idx in range(args.start, min(len(dataset), args.start + args.samples * args.stride), args.stride):
        seed(303 + idx)
        old = baseline[idx]
        seed(303 + idx)
        new = dataset[idx]
        if old is None or new is None:
            assert old is None and new is None
            rows.append(dict(index=idx, skipped=True))
            continue
        for key in ('past_k_time_diffs', 'object_bbx_center', 'pairwise_t_matrix'):
            np.testing.assert_array_equal(old['ego'][key], new['ego'][key])
        history = new['ego']['ectra_ego_history']
        batch = train_utils.to_device(dataset.collate_batch_test([new]), torch.device('cuda'))
        model.zero_grad()
        if probe is not None:
            probe.clear()
        result = model(batch['ego'], dataset)
        detection_loss = criterion(result, batch['ego']['label_dict'])
        loss = add_auxiliary_losses(cfg, result, detection_loss)
        assert torch.isfinite(loss).all()
        input_gradients = None if probe is None else {
            'detection': probe.summarize(detection_loss),
            'total': probe.summarize(loss)}
        loss.backward()
        grads = {}
        for name in ('motion_net', 'calib_net', 'trust_net', 'candidate_net'):
            grads[name] = sum(float(p.grad.detach().abs().sum()) for p in getattr(model.ectra, name).parameters() if p.grad is not None)
        grads['roi_refiner'] = sum(float(p.grad.detach().abs().sum()) for p in model.ectra_roi.parameters() if p.grad is not None)
        assert all(np.isfinite(v) for v in grads.values())
        assert all(p.grad is None for p in model.pillar_vfe.parameters())
        with torch.no_grad():
            boxes, scores, gt = dataset.post_process(batch, {'ego': result})
        row = dict(index=idx, loss=float(loss.detach()), gradients=grads,
                   history_valid=history['valid'], history_age_ms=history['age_ms'],
                   history_ids=history['frame_ids'], boxes=0 if boxes is None else len(boxes),
                   diagnostics={key: float(value.detach().mean()) for key, value in result.items()
                                if key.startswith('ectra_') and torch.is_tensor(value)})
        rows.append(row)
        if input_gradients is not None:
            row['input_gradients'] = input_gradients
        print(json.dumps(row), flush=True)
        del result, loss, detection_loss, batch
        if probe is not None:
            probe.clear()
    valid_rows = [row for row in rows if not row.get('skipped')]
    assert valid_rows, 'No usable audit samples'
    assert all(any(row['gradients'][key] > 0 for row in valid_rows) for key in valid_rows[0]['gradients'])
    destination.write_text(json.dumps(dict(status='passed', arguments=vars(args),
                                           model_mode='train_without_optimizer_step',
                                           rows=rows), indent=2))
    if probe is not None:
        probe.close()


if __name__ == '__main__':
    main()

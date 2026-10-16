"""Real batch: exact identity fusion, trainable residual, immutable detector/BN."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.utils.e3dsnn_fusion_probe import detector_digest


def run(config, checkpoint):
    random.seed(20260929)
    np.random.seed(20260929)
    torch.manual_seed(20260929)
    cfg = load_yaml(config)
    dataset = build_dataset(cfg, train=True)
    batch = train_utils.to_device(dataset.collate_batch_train([dataset[0], dataset[1]]), torch.device('cuda'))
    model = train_utils.create_model(cfg).cuda()
    assert model.freeze_detector and cfg['model']['args']['identity_fusion_init']
    state = torch.load(checkpoint, map_location='cpu')
    model.load_single(state)
    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in state.items())
    model.train()
    assert not model.backbone_3d.training and not model.backbone_2d.training
    before = detector_digest(model)
    original_encoder = model.encode_lidar
    data = batch['ego']
    with torch.no_grad():
        ego = original_encoder(data['processed_lidar'], 2)
        road = original_encoder(data['road_lidar'], 2)
    # Reuse frozen feature tensors to exclude independent sparse-CUDA roundoff.
    model.encode_lidar = lambda lidar, b: ego if lidar is data['processed_lidar'] else road
    spatial = ego.dense().flatten(1, 2)
    detector = model.detect_bev
    captured = []
    def capture(value):
        captured.append(torch.equal(value, spatial))
        return detector(value)
    model.detect_bev = capture
    with torch.no_grad():
        baseline = detector(spatial)
        initial = model(data)
        assert captured == [True], 'Initial fusion changed ego features'
        for key in ('psm', 'rm', 'dm'):
            torch.testing.assert_close(initial[key], baseline[key], atol=1e-6, rtol=1e-6)
    model.detect_bev = detector
    criterion = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    losses = []
    last_before = model.fusion_net[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, criterion, batch)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        grad = model.fusion_net[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert not torch.equal(last_before, model.fusion_net[-1].weight)
    assert detector_digest(model) == before
    model.eval(); model.train()
    assert not model.backbone_3d.training and not model.backbone_2d.training
    return dict(passed=True, initial_feature_identity_exact=True, initial_heads_match=True,
        frozen_weights_and_buffers_unchanged=True, fusion_weight_updated=True, losses=losses,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        detector_sha256=before, scope='Two real augmented train frames, two optimizer steps; smoke only, weights discarded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config, args.checkpoint)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)

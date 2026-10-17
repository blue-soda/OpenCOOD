"""Real-data warmup/joint gradient and fixed-BN checks for auxiliary controls."""
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
from opencood.utils.e3dsnn_fusion_probe import detector_digest, bn_digest


def run(config, checkpoint):
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    cfg = load_yaml(config)
    dataset = build_dataset(cfg, train=True)
    batch = train_utils.to_device(dataset.collate_batch_train([dataset[0], dataset[1]]), torch.device('cuda'))
    model = train_utils.create_model(cfg).cuda()
    model.load_single(torch.load(checkpoint, map_location='cpu'))
    assert model.enable_ego_aux and model.freeze_detector_bn and not model.freeze_detector
    criterion = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    initial_bn = bn_digest(model)
    report = dict(config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        ego_aux_weight=model.ego_aux_weight, phases={})
    for phase in ('warmup', 'joint'):
        model.train()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(phase == 'joint' or name.startswith('fusion_net.') or name == 'fusion_scale')
        assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        detector_before = detector_digest(model)
        optimizer.zero_grad(set_to_none=True)
        output, loss = checked_forward(model, criterion, batch)
        parts = model.last_loss_components
        assert abs(float(loss.detach()) - (parts['fused_detection_loss'] + model.ego_aux_weight * parts['ego_detection_loss'])) < 1e-5
        loss.backward()
        groups = {}
        for prefix in ('fusion_net.', 'backbone_3d.', 'backbone_2d.', 'reg_head.'):
            gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert all(torch.isfinite(g).all() for g in gradients)
            groups[prefix] = sum(float(g.abs().sum()) for g in gradients)
            if phase == 'joint' or prefix == 'fusion_net.':
                assert groups[prefix] > 0, (phase, prefix)
            else:
                assert not gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if phase == 'warmup':
            assert detector_digest(model) == detector_before
        else:
            assert detector_digest(model) != detector_before
        assert bn_digest(model) == initial_bn
        report['phases'][phase] = dict(loss=float(loss.detach()), components=parts,
            gradient_l1=groups, bn_unchanged=True)
    model.eval()
    with torch.no_grad():
        assert 'ego_output' not in model(batch['ego'])
        if model.fusion_readout == 'continuous':
            data = batch['ego']
            original_encoder = model.encode_lidar
            size = data['object_bbx_mask'].shape[0]
            cached = {id(data[key]): original_encoder(data[key], size)
                      for key in ('processed_lidar', 'road_lidar')}
            model.encode_lidar = lambda lidar, batch_size: cached[id(lidar)]
            model.packet_roundtrip = True
            model.collect_fusion_diagnostics = True
            continuous = model(data)
            model.fusion_readout = 'activated'
            discrete = model(data)
            assert continuous['packet_bytes'] == discrete['packet_bytes']
            assert torch.isfinite(continuous['psm']).all() and torch.isfinite(continuous['rm']).all()
            assert not torch.equal(continuous['psm'], discrete['psm'])
            assert 'above_cap_fraction' not in continuous['fusion_probe']
            report['readout_audit'] = dict(sender_packets_identical=True,
                packet_bytes=continuous['packet_bytes'], distinct_detector_output=True,
                cached_encoder_features=True, continuous_probe_omits_count_cap=True)
            model.encode_lidar = original_encoder
            model.fusion_readout = 'continuous'
    report.update(passed=True, inference_has_no_auxiliary_branch=True,
                  scope='Two real train frames; one warmup and one joint optimizer step; weights discarded')
    return report


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

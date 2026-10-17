"""Build four donor compositions and audit native equivalence on real data."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.utils.e3dsnn_transplant import compose_state, encoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest


def audit(root, source):
    root, source = Path(root), Path(source)
    repo = Path(__file__).resolve().parents[2]
    config_dir = repo / 'opencood/hypes_yaml/dair-v2x/snn'
    summary = json.loads((source / 'aux_final_summary.json').read_text())
    assert summary['passed']
    states, donors = {}, {}
    for kind, prefix in [('s', 'count4'), ('a', 'relu')]:
        path = source / (prefix + '_aux1_joint30') / 'best_ap70.pth'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == summary['runs'][prefix + '_aux1_joint30']['best_checkpoint']['sha256']
        states[kind] = torch.load(str(path), map_location='cpu')
        donors[kind] = dict(path=str(path), sha256=sha)
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    cfg = load_yaml(str(config_dir / 'e3dsnn_hybrid_ss_181.yaml'))
    dataset = build_dataset(cfg, train=True)
    batch = train_utils.to_device(dataset.collate_batch_train([dataset[0], dataset[1]]), torch.device('cuda'))
    data = batch['ego']; size = data['object_bbx_mask'].shape[0]
    report = dict(donors=donors, combinations={}, scope='Two real augmented training frames; '
        'native output equality uses cached encoder features; optimizer updates discarded.')
    cached_encoders, packets = {}, {}
    for name in ('ss', 'sa', 'as', 'aa'):
        config = config_dir / ('e3dsnn_hybrid_' + name + '_181.yaml')
        cfg = load_yaml(str(config)); model = train_utils.create_model(cfg).cuda()
        state = compose_state(states[name[0]], states[name[1]])
        model.load_state_dict(state, strict=True)
        checkpoint = root / (name + '_initial.pth')
        if checkpoint.exists():
            old = torch.load(str(checkpoint), map_location='cpu')
            assert set(old) == set(state) and all(torch.equal(old[k], state[k]) for k in state)
        else:
            torch.save(state, str(checkpoint))
        model.eval(); original_encode = model.encode_lidar
        with torch.no_grad():
            if name[0] not in cached_encoders:
                cached_encoders[name[0]] = {id(data[k]): original_encode(data[k], size)
                    for k in ('processed_lidar', 'road_lidar')}
            cached = cached_encoders[name[0]]
            model.encode_lidar = lambda lidar, batch_size: cached[id(lidar)]
            model.packet_roundtrip = True
            result = model(data)
            assert all(torch.isfinite(result[k]).all() for k in ('psm', 'rm', 'dm'))
            if name[0] in packets: assert packets[name[0]] == result['packet_bytes']
            packets[name[0]] = result['packet_bytes']
            native_equal = None
            if name in ('ss', 'aa'):
                native_cfg = load_yaml(str(config_dir / ('e3dsnn_pair_' + ('count4' if name[0] == 's' else 'relu') + '_aux1_181.yaml')))
                native = train_utils.create_model(native_cfg).cuda().eval()
                native.load_state_dict(states[name[0]], strict=True)
                # Independent encoder equality also checks selective activations.
                direct = native.encode_lidar(data['processed_lidar'], size)
                expected = cached[id(data['processed_lidar'])]
                def canonical(sparse):
                    idx = sparse.indices.long(); d, h, w = sparse.spatial_shape
                    order = (((idx[:, 0] * d + idx[:, 1]) * h + idx[:, 2]) * w + idx[:, 3]).argsort()
                    return sparse.indices[order], sparse.features[order]
                actual_indices, actual_values = canonical(direct)
                expected_indices, expected_values = canonical(expected)
                assert torch.equal(actual_indices, expected_indices)
                encoder_max_error = float((actual_values - expected_values).abs().max())
                torch.testing.assert_close(actual_values, expected_values, rtol=1e-5, atol=1e-6)
                native.encode_lidar = lambda lidar, batch_size: cached[id(lidar)]
                native.packet_roundtrip = True
                reference = native(data)
                assert reference['packet_bytes'] == result['packet_bytes']
                assert all(torch.equal(reference[k], result[k]) for k in ('psm', 'rm', 'dm'))
                native_equal = True
                del native, reference, direct
        model.encode_lidar = original_encode; model.packet_roundtrip = False
        model.train(); before = encoder_digest(model); bn_before = bn_digest(model)
        assert not model.backbone_3d.training
        criterion = train_utils.create_loss(cfg).cuda()
        optimizer = train_utils.setup_optimizer(cfg, model)
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, criterion, batch); loss.backward()
        assert all(p.grad is None for p in model.backbone_3d.parameters())
        gradients = {}
        for prefix in ('fusion_net.', 'backbone_2d.', 'reg_head.'):
            values = [p.grad for k, p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values)
            gradients[prefix] = sum(float(v.abs().sum()) for v in values)
        assert gradients['reg_head.'] > 0 and sum(gradients.values()) > 0
        receiver_before = model.reg_head.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        assert encoder_digest(model) == before and bn_digest(model) == bn_before
        assert not torch.equal(model.reg_head.weight, receiver_before)
        report['combinations'][name] = dict(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), encoder_sha256=before,
            packet_bytes=result['packet_bytes'], native_exact_output_match=native_equal,
            native_encoder_max_abs_error=encoder_max_error if native_equal else None,
            encoder_frozen_after_step=True, bn_fixed_after_step=True, gradient_l1=gradients,
            loss=float(loss.detach()), encoder_activation=model.encoder_activation,
            receiver_activation=model.receiver_activation)
        del model, optimizer, criterion, result, loss, original_encode
        gc.collect(); torch.cuda.empty_cache()
    report['passed'] = True
    with (root / 'audit.json').open('x') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    audit(args.root, args.source)

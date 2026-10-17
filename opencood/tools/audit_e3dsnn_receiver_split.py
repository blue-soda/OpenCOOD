"""Real-data audit before any receiver-factorial evaluation or training."""
import argparse
import gc
import json
from pathlib import Path
import random
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward
from opencood.tools.launch_e3dsnn_receiver_round import source_gate, COMBINATIONS
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, sha
from opencood.utils.e3dsnn_receiver_split import compose_receiver_state
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest


def audit(root, source):
    root, source = Path(root), Path(source)
    summary = source_gate(source)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'audit.json').exists(): raise FileExistsError('Audit already completed')
    configs = REPO / 'opencood/hypes_yaml/dair-v2x/snn'
    states, donors = {}, {}
    for kind, combo in [('s', 'ss'), ('a', 'sa')]:
        path = source / ('hybrid_' + combo + '_adapt10') / 'best_ap70.pth'
        assert sha(path) == summary['combinations'][combo]['checkpoint_sha256']
        states[kind] = torch.load(str(path), map_location='cpu')
        donors[kind] = dict(path=str(path), sha256=sha(path), source_combination=combo)
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    cfg = load_yaml(str(configs / 'e3dsnn_receiver_ss_181.yaml'))
    dataset = build_dataset(cfg, train=True)
    batch = train_utils.to_device(dataset.collate_batch_train([dataset[0], dataset[1]]), torch.device('cuda'))
    data = batch['ego']; size = data['object_bbx_mask'].shape[0]
    report = dict(source_summary_sha256=sha(source / 'adapt10_summary.json'), donors=donors,
                  combinations={}, scope='Two real augmented training frames. Native cached-output equality; '
                  'packet versus direct feature equality; finite gradients. Optimizer updates discarded.')
    cached, packet_lengths = None, None
    for combo in COMBINATIONS:
        config = configs / ('e3dsnn_receiver_' + combo + '_181.yaml')
        cfg = load_yaml(str(config)); model = train_utils.create_model(cfg).cuda()
        state = compose_receiver_state(states['s'], states['a'], combo[0], combo[1])
        model.load_state_dict(state, strict=True)
        target = root / (combo + '_initial.pth')
        if target.exists():
            old = torch.load(str(target), map_location='cpu')
            assert set(old) == set(state) and all(torch.equal(old[k], state[k]) for k in state)
        else: torch.save(state, str(target))
        model.eval(); original_encode = model.encode_lidar
        # Check module types against the corresponding adapted native receiver.
        for kind in ('s', 'a'):
            donor_combo = 'ss' if kind == 's' else 'sa'
            native_cfg = load_yaml(str(configs / ('e3dsnn_hybrid_' + donor_combo + '_181.yaml')))
            native = train_utils.create_model(native_cfg).cuda().eval()
            native.load_state_dict(states[kind], strict=True)
            for part, selected in [('fusion_net', combo[0]), ('fusion_activation', combo[0]),
                                   ('backbone_2d', combo[1]), ('cls_head', combo[1]),
                                   ('reg_head', combo[1]), ('dir_head', combo[1]), ('backbone_3d', kind)]:
                if selected == kind:
                    assert [type(m) for m in getattr(model, part).modules()] == [type(m) for m in getattr(native, part).modules()]
            with torch.no_grad():
                if cached is None:
                    cached = {id(data[k]): native.encode_lidar(data[k], size) for k in ('processed_lidar', 'road_lidar')}
                if combo == kind * 2:
                    native.encode_lidar = lambda lidar, n: cached[id(lidar)]
                    native.packet_roundtrip = True
                    reference = native(data)
            del native
        with torch.no_grad():
            model.encode_lidar = lambda lidar, n: cached[id(lidar)]
            model.packet_roundtrip = True
            model.collect_fusion_diagnostics = True
            result = model(data)
            assert all(torch.isfinite(result[k]).all() for k in ('psm', 'rm', 'dm'))
            if packet_lengths is None: packet_lengths = result['packet_bytes']
            assert packet_lengths == result['packet_bytes']
            model.packet_roundtrip = False; model.collect_fusion_diagnostics = False
            direct = model(data)
            assert all(torch.equal(result[k], direct[k]) for k in ('psm', 'rm', 'dm'))
            native_equal = None
            if combo in ('ss', 'aa'):
                assert reference['packet_bytes'] == result['packet_bytes']
                assert all(torch.equal(reference[k], result[k]) for k in ('psm', 'rm', 'dm'))
                native_equal = True
                del reference
        model.encode_lidar = original_encode
        model.train(); before = encoder_digest(model); bn_before = bn_digest(model)
        assert not model.backbone_3d.training
        criterion = train_utils.create_loss(cfg).cuda()
        optimizer = train_utils.setup_optimizer(cfg, model)
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, criterion, batch); loss.backward()
        assert all(p.grad is None for p in model.backbone_3d.parameters())
        grads = {}
        for prefix in ('fusion_net.', 'backbone_2d.', 'reg_head.'):
            values = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values)
            grads[prefix] = sum(float(v.abs().sum()) for v in values)
            assert grads[prefix] > 0
        head_before = model.reg_head.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        assert encoder_digest(model) == before and bn_digest(model) == bn_before
        assert before == summary['combinations']['ss']['frozen_encoder.json']['current_sha256']
        assert not torch.equal(model.reg_head.weight, head_before)
        report['combinations'][combo] = dict(checkpoint_sha256=sha(target), encoder_sha256=before,
            config_sha256={suffix: sha(configs / ('e3dsnn_receiver_' + combo + suffix))
                           for suffix in ('_181.yaml', '_ego_181.yaml')},
            native_exact_output_match=native_equal, packet_and_direct_outputs_equal=True,
            packet_bytes=packet_lengths, encoder_frozen_after_step=True, bn_fixed_after_step=True,
            gradient_l1=grads, loss=float(loss.detach()), fusion_activation=model.fusion_activation_kind,
            bev_activation=model.bev_activation)
        del model, optimizer, criterion, result, direct, loss, original_encode
        gc.collect(); torch.cuda.empty_cache()
    report['passed'] = True
    with (root / 'audit.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    audit(args.root, args.source)

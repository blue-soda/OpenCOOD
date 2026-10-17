"""Check unit-step identity, scale gradients and unchanged sender on real data."""
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
from opencood.tools.launch_e3dsnn_bevscale_round import GROUPS, source_gate
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, sha
from opencood.models.sub_modules.e3dsnn.bev_scale import add_unit_scales, ScaledCount4
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_fusion_probe import bn_digest


def audit(root, source):
    root, source = Path(root), Path(source)
    initializer, source_entry = source_gate(source)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'audit.json').exists(): raise FileExistsError('Audit already completed')
    configs = REPO / 'opencood/hypes_yaml/dair-v2x/snn'
    donor = torch.load(str(initializer), map_location='cpu')
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    cfg = load_yaml(str(configs / 'e3dsnn_bevscale_fixed_high_181.yaml'))
    dataset = build_dataset(cfg, train=True)
    batch = train_utils.to_device(dataset.collate_batch_train([dataset[0], dataset[1]]), torch.device('cuda'))
    data = batch['ego']; size = data['object_bbx_mask'].shape[0]
    native = train_utils.create_model(load_yaml(str(configs / 'e3dsnn_hybrid_ss_181.yaml'))).cuda().eval()
    native.load_state_dict(donor, strict=True)
    with torch.no_grad():
        cached = {id(data[k]): native.encode_lidar(data[k], size) for k in ('processed_lidar', 'road_lidar')}
        native.encode_lidar = lambda lidar, n: cached[id(lidar)]
        native.packet_roundtrip = True
        reference = native(data)
    report = dict(source_summary_sha256=sha(source / 'adapt10_summary.json'), initializer=str(initializer),
                  initializer_sha256=sha(initializer), encoder_sha256=encoder_digest(native), groups={},
                  scope='Two real augmented frames; cached encoder identity, packet roundtrip, one discarded update.')
    del native
    for group in GROUPS:
        config = configs / ('e3dsnn_bevscale_' + group + '_181.yaml')
        cfg = load_yaml(str(config)); model = train_utils.create_model(cfg)
        initial = add_unit_scales(donor, model.state_dict())
        assert all(torch.equal(initial[k], v) for k, v in donor.items())
        checkpoint = root / 'initial.pth'
        if checkpoint.exists():
            old = torch.load(str(checkpoint), map_location='cpu')
            assert set(old) == set(initial) and all(torch.equal(old[k], v) for k, v in initial.items())
        else: torch.save(initial, str(checkpoint))
        model.load_state_dict(initial, strict=True); model = model.cuda().eval()
        assert model.bev_scale_report()['changed_from_one'] == 0
        original_encode = model.encode_lidar
        model.encode_lidar = lambda lidar, n: cached[id(lidar)]
        with torch.no_grad():
            model.packet_roundtrip = True
            result = model(data)
            assert result['packet_bytes'] == reference['packet_bytes']
            assert all(torch.equal(result[k], reference[k]) for k in ('psm', 'rm', 'dm'))
            model.packet_roundtrip = False
            direct = model(data)
            assert all(torch.equal(result[k], direct[k]) for k in ('psm', 'rm', 'dm'))
        model.encode_lidar = original_encode; model.train()
        before, bn_before = encoder_digest(model), bn_digest(model)
        criterion = train_utils.create_loss(cfg).cuda(); optimizer = train_utils.setup_optimizer(cfg, model)
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, criterion, batch); loss.backward()
        assert all(p.grad is None for p in model.backbone_3d.parameters())
        grads = {}
        for prefix in ('fusion_net.', 'backbone_2d.', 'reg_head.'):
            values = [p.grad for k, p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values)
            grads[prefix] = sum(float(v.abs().sum()) for v in values); assert grads[prefix] > 0
        scales = [m.log_scale for m in model.backbone_2d.modules() if isinstance(m, ScaledCount4)]
        if group.startswith('learned'):
            assert all(s.requires_grad and s.grad is not None and torch.isfinite(s.grad).all() for s in scales)
            scale_gradient = sum(float(s.grad.abs()) for s in scales); assert scale_gradient > 0
        else:
            assert all(not s.requires_grad and s.grad is None for s in scales)
            scale_gradient = 0.
        head_before = model.reg_head.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True); optimizer.step()
        assert encoder_digest(model) == before == report['encoder_sha256']
        assert before == source_entry['frozen_encoder.json']['current_sha256']
        assert bn_digest(model) == bn_before
        assert not torch.equal(model.reg_head.weight, head_before)
        scale_state = model.bev_scale_report()
        assert (scale_state['changed_from_one'] > 0) == group.startswith('learned')
        report['groups'][group] = dict(config_sha256={suffix: sha(configs / ('e3dsnn_bevscale_' + group + suffix))
              for suffix in ('_181.yaml', '_ego_181.yaml')}, native_exact_output_match=True,
              packet_direct_output_match=True, packet_bytes=result['packet_bytes'],
              encoder_and_bn_fixed=True, gradient_l1=grads, scale_gradient_l1=scale_gradient,
              scales_after_discarded_step=scale_state, loss=float(loss.detach()),
              parameters=sum(p.numel() for p in model.parameters()), lr=cfg['optimizer']['lr'])
        del model, criterion, optimizer, result, direct, loss, original_encode
        gc.collect(); torch.cuda.empty_cache()
    report['initial_state_sha256'] = sha(root / 'initial.pth'); report['passed'] = True
    with (root / 'audit.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--source', required=True)
    args = parser.parse_args(); audit(args.root, args.source)

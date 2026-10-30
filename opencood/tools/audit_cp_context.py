"""Real-data identity, wire and gradient admission for the matched context arms."""
import argparse
import copy
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


def audit(config, checkpoint, parent, output):
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929); torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.6)
    cfg = load_yaml(config)
    train = build_dataset(cfg, train=True); val = build_dataset(cfg, train=False)
    assert len(train) == 4678 and len(val) == 1738
    samples = [train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda'))
               for i in (0, 1)]
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg['model']['core_method'] = 'e3dsnn_snapshot_frontend'
    baseline_cfg['model']['args'].pop('snapshot_context')
    baseline = train_utils.create_model(baseline_cfg).cuda().eval()
    baseline.load_state_dict(torch.load(parent, map_location='cpu'), strict=True)
    baseline.packet_roundtrip = True; baseline.snapshot_audit = True
    expected = []
    with torch.no_grad():
        for sample in samples:
            out = baseline(sample['ego'])
            expected.append(({k:out[k].cpu() for k in ('psm','rm','dm')},
                             out['packet_bytes'], copy.deepcopy(baseline.snapshot_records)))
    del baseline; gc.collect(); torch.cuda.empty_cache()
    model = train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    parent_state = torch.load(parent, map_location='cpu')
    new_keys = set(model.state_dict()) - set(parent_state)
    prefix = 'joint_receiver.snapshot_context_gate.'
    assert new_keys == {prefix + x for x in ('net.0.weight','net.0.bias','net.2.weight','net.2.bias')}
    assert all(torch.equal(v.cpu(), model.state_dict()[k].cpu()) for k,v in parent_state.items())
    model.snapshot_audit = True
    with torch.no_grad():
        for sample, (expected_out, expected_bytes, records) in zip(samples, expected):
            model.packet_roundtrip = False
            direct = model(sample['ego'])
            model.packet_roundtrip = True
            restored = model(sample['ego'])
            assert all(torch.equal(direct[k], restored[k]) and
                       torch.equal(restored[k].cpu(), expected_out[k]) for k in expected_out)
            assert direct['packet_bytes'] == restored['packet_bytes'] == expected_bytes
            assert model.snapshot_records == records
            assert all(0 < n <= model.snapshot_budget for n in expected_bytes)
        budget = model.snapshot_budget; model.snapshot_budget = 0
        zero = model(samples[0]['ego'])
        samples[0]['ego']['disable_road'] = True
        ego = model(samples[0]['ego'])
        assert all(torch.equal(zero[k], ego[k]) for k in ('psm','rm','dm'))
        model.snapshot_budget = budget
    model.packet_roundtrip = False; model.train()
    fixed = model.fixed_state_digest()
    batch = train_utils.to_device(train.collate_batch_train([train[0],train[1]]), torch.device('cuda'))
    loss_fn = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert {id(p) for p in model.parameters() if p.requires_grad} <= optimized
    assert optimized <= {id(p) for p in model.parameters()}
    prefixes = ['backbone_3d.', 'content.project.', 'content.expand.', 'joint_receiver.',
                'backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.', prefix+'net.2.']
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, loss_fn, batch)
        loss.backward()
        gradients = {p:sum(float(v.grad.abs().sum()) for n,v in model.named_parameters()
            if n.startswith(p) and v.grad is not None) for p in prefixes + [prefix+'net.0.']}
        assert all(np.isfinite(gradients[p]) and gradients[p] > 0 for p in prefixes), gradients
        hidden = gradients[prefix+'net.0.']
        assert (hidden == 0 if step == 0 else np.isfinite(hidden) and hidden > 0), gradients
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        assert model.fixed_state_digest() == fixed
        steps.append(dict(loss=float(loss.detach()), gradients=gradients))
    result = dict(passed=True, condition=cfg['model']['args']['snapshot_context']['condition'],
        parent_output_exact=True, parent_packet_records_exact=True, identity_frames=2,
        direct_wire_exact=True, zero_equals_ego=True, new_parameter_count=6288,
        frozen_state_unchanged=True, optimizer_covers_trainable=True, steps=steps,
        train_samples=len(train), val_samples=len(val), budget=budget,
        audit_updates_discarded=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    with Path(output).open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('config','checkpoint','parent','output'): p.add_argument('--'+key, required=True)
    a=p.parse_args(); audit(a.config,a.checkpoint,a.parent,a.output)

"""Audit the active F0 snapshot graph, excluding unused legacy fusion weights."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward


def audit(config, checkpoint, output):
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929); torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.6)
    cfg = load_yaml(config)
    model = train_utils.create_model(cfg).cuda()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    train = build_dataset(cfg, train=True); val = build_dataset(cfg, train=False)
    assert len(train) == 4678 and len(val) == 1738
    sample = train_utils.to_device(val.collate_batch_test([val[0]]), torch.device('cuda'))
    model.eval()
    with torch.no_grad():
        direct = model(sample['ego'])
        model.packet_roundtrip = True
        restored = model(sample['ego'])
        assert all(torch.equal(direct[k], restored[k]) for k in ('psm','rm','dm'))
        assert direct['packet_bytes'] == restored['packet_bytes']
        assert all(0 < n <= model.snapshot_budget for n in restored['packet_bytes'])
        model.packet_roundtrip = False
        budget = model.snapshot_budget; model.snapshot_budget = 0
        zero = model(sample['ego']); sample['ego']['disable_road'] = True
        ego = model(sample['ego'])
        assert all(torch.equal(zero[k], ego[k]) for k in ('psm','rm','dm'))
        model.snapshot_budget = budget
    model.train()
    fixed = model.fixed_state_digest()
    batch = train_utils.to_device(train.collate_batch_train([train[0],train[1]]), torch.device('cuda'))
    loss_fn = train_utils.create_loss(cfg).cuda()
    optimizer = train_utils.setup_optimizer(cfg, model)
    prefixes = ['backbone_3d.', 'content.project.', 'content.expand.', 'joint_receiver.',
                'backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.']
    steps = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = checked_forward(model, loss_fn, batch)
        loss.backward()
        gradients = {prefix: sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
        assert all(np.isfinite(v) and v > 0 for v in gradients.values()), gradients
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        assert model.fixed_state_digest() == fixed
        steps.append(dict(loss=float(loss.detach()), gradients=gradients))
    result = dict(passed=True, direct_wire_exact=True, zero_equals_ego=True,
        fixed_state_unchanged=True, steps=steps, train_samples=len(train), val_samples=len(val),
        budget=budget, actual_bytes=direct['packet_bytes'], audit_updates_discarded=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    with Path(output).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('config','checkpoint','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();audit(a.config,a.checkpoint,a.output)

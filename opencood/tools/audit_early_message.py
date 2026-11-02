"""Admission of a paid-message gradient through the existing SNN deep path."""
import argparse
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_shallow_message import run as shallow_admission
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import checked_forward, write_json


def run(a):
    shallow_admission(a)
    seed(); cfg = load_yaml(a.config)
    model = train_utils.create_model(cfg).cuda()
    initial = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(initial, strict=True); model.train()
    model.message_dropout = 0.  # This discarded gradient probe requires a message.
    train = build_dataset(cfg, train=True)
    batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
    criterion = train_utils.create_loss(cfg).cuda()
    # Block the alternative detail gradient; P/D gradients must now travel
    # through conv4/conv_out and the ANN BEV. No optimizer step is performed.
    h = model.local_detail.register_forward_hook(lambda m, i, o: o.detach())
    try:
        with no_codec():
            output, loss = checked_forward(model, criterion, batch); loss.backward()
    finally:
        h.remove()
    gradients = {}
    for n, p in model.named_parameters():
        if n.startswith(('content.', 'joint_receiver.')):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
            gradients[n] = float(p.grad.abs().sum())
    assert all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items())
    model.eval(); val = build_dataset(cfg, train=False); differences = []
    captured = []
    h = model.backbone_2d.register_forward_pre_hook(
        lambda m, i: captured.append(i[0]['spatial_features'].detach().clone()))
    try:
        for i in (0, 1):
            batch = train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda'))
            captured.clear()
            with no_codec(), torch.no_grad():
                model(batch['ego'])
                model(dict(batch['ego'], disable_road=True))
            assert len(captured) == 2
            differences.append(float((captured[0]-captured[1]).abs().max()))
            assert model._ego_indices is None
    finally:
        h.remove()
    assert any(x > 0 for x in differences)
    assert all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items())
    write_json(Path(a.output).with_name('deep_admission.json'), dict(passed=True,
        detail_gradient_detached=True, paid_interface_gradient_l1=gradients,
        deep_count_change_max=differences, state_unchanged=True, no_optimizer=True,
        no_codec=True, no_AP=True, parameter_count=sum(p.numel() for p in model.parameters())))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'source', 'output'): p.add_argument('--'+k, required=True)
    run(p.parse_args())

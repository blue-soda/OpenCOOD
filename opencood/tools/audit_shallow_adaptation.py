"""Real-input gradient admission for same-range shallow count adaptation."""
import argparse
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec, run as quota_audit
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer


def run(a):
    # Includes explicit two-frame, three-quota wire parity; discarded training
    # updates and all following gradient checks execute with codec prohibited.
    quota_audit(a)
    seed(); cfg = load_yaml(a.config)
    model = train_utils.create_model(cfg).cuda()
    initial = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(initial, strict=True); model.train()
    assert model.precision_mode in ('count4', 'count8', 'count16')
    frozen = model.fixed_state_digest()
    train = build_dataset(cfg, train=True)
    batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
    optimizer = make_optimizer(cfg, model); criterion = train_utils.create_loss(cfg).cuda()
    observed = []
    def capture(module, inputs, output):
        inputs[0].retain_grad(); observed.append(inputs[0])
    h = model.shallow_readout.register_forward_hook(capture)
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True); observed.clear()
        with no_codec():
            out, loss = checked_forward(model, criterion, batch); loss.backward()
        assert not out['communication_measured'] and 'packet_bytes' not in out
        assert observed and all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in observed)
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith('local_detail.'))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); assert torch.isfinite(norm)
        optimizer.step(); assert model.fixed_state_digest() == frozen
        rows.append(dict(step=step+1, loss=float(loss), norm=float(norm),
                         readout_input_gradient_l1=sum(float(x.grad.abs().sum()) for x in observed)))
    h.remove()
    assert any(not torch.equal(v.cpu(), initial[n]) for n,v in model.state_dict().items() if n.startswith('local_detail.'))
    write_json(Path(a.output).with_name('gradient_admission.json'), dict(passed=True,
        mode=model.precision_mode, steps=rows, frozen_state_unchanged=True,
        no_codec=True, updates_discarded=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'output'): p.add_argument('--'+k, required=True)
    run(p.parse_args())

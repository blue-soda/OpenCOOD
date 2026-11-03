"""Admission of the serial packed-BEV alternative; no task AP claim."""
import argparse
import copy
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def observe(model, samples):
    fused = []; macs = []; total = [0]
    original = model.detect_bev_sample
    def capture(spatial, index):
        fused.append(digest(spatial))
        return original(spatial, index)
    def count(module, inputs, output):
        if isinstance(module, torch.nn.ConvTranspose2d):
            total[0] += inputs[0].numel() * module.out_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        else:
            total[0] += output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
    model.detect_bev_sample = capture
    handles = [m.register_forward_hook(count) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    try:
        for sample in samples:
            total[0] = 0
            with no_codec(), torch.no_grad(): model(sample['ego'])
            macs.append(total[0])
    finally:
        model.detect_bev_sample = original
        for h in handles: h.remove()
    return dict(fused_sha256=fused, dense_2d_MACs=macs,
                total_parameters=sum(p.numel() for p in model.parameters()))


def run(a):
    seed(); torch.set_num_threads(1)
    cfg = load_yaml(a.config)
    val = build_dataset(cfg, train=False); train = build_dataset(cfg, train=True)
    samples = [train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda')) for i in (0, 1)]
    source = torch.load(a.source, map_location='cpu')
    base_cfg = copy.deepcopy(cfg); base_cfg['model']['core_method'] = 'e3dsnn_snapshot_early_message'
    base = train_utils.create_model(base_cfg).cuda(); base.load_state_dict(source, strict=True); base.eval()
    reference = observe(base, samples)
    del base
    model = train_utils.create_model(cfg).cuda()
    initial = torch.load(a.checkpoint, map_location='cpu'); model.load_state_dict(initial, strict=True); model.eval()
    removed = ('backbone_3d.conv4.', 'backbone_3d.conv_out.', 'local_detail.')
    assert {k for k in initial if not k.startswith('serial_project.')} == {k for k in source if not k.startswith(removed)}
    assert all(torch.equal(v, source[k]) for k, v in initial.items() if not k.startswith('serial_project.'))
    assert model.local_detail is None and model.backbone_3d.conv4 is None and model.backbone_3d.conv_out is None
    check = torch.arange(64*8*10, device='cuda', dtype=torch.float32).reshape(1,64,8,10) % 5
    packed = F.pixel_unshuffle(check, 2)
    assert torch.equal(F.pixel_shuffle(packed, 2), check)
    with torch.no_grad(): assert torch.equal(model.serial_project(packed)[:, :256], packed)
    candidate = observe(model, samples)
    assert candidate['fused_sha256'] == reference['fused_sha256']
    # The retained backend processes its original spatial grid. Only dense 2D
    # MACs are counted; sparse 3D work, other ops, latency and energy are excluded.
    assert all(c <= r for c, r in zip(candidate['dense_2d_MACs'], reference['dense_2d_MACs']))
    frames = []
    for i, sample in enumerate(samples):
        for budget in (0, 2304, 8192):
            model.snapshot_budget = budget
            with no_codec(), torch.no_grad(): direct = model(sample['ego'])
            assert not direct['communication_measured'] and 'packet_bytes' not in direct
            model.packet_roundtrip = True
            with torch.no_grad(): wire = model(sample['ego'])
            model.packet_roundtrip = False
            assert all(torch.equal(direct[k], wire[k]) for k in ('psm', 'rm', 'dm'))
            if budget == 0:
                with no_codec(), torch.no_grad(): ego = model(dict(sample['ego'], disable_road=True))
                assert all(torch.equal(direct[k], ego[k]) for k in ('psm', 'rm', 'dm'))
            frames.append(dict(index=i, nominal_budget=budget, actual_bytes=wire['packet_bytes'], wire_equal=True))
    assert all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items())
    model.snapshot_budget = 8192; model.train(); model.message_dropout = 0.
    frozen = model.fixed_state_digest()
    batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
    criterion = train_utils.create_loss(cfg).cuda(); optimizer = make_optimizer(cfg, model)
    prefixes = ('backbone_3d.conv3.', 'content.project.', 'content.expand.', 'joint_receiver.',
                'serial_project.', 'backbone_2d.', 'cls_head.', 'reg_head.')
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with no_codec():
            output, loss = checked_forward(model, criterion, batch); loss.backward()
        assert not output['communication_measured'] and 'packet_bytes' not in output
        grads = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        for prefix in prefixes:
            assert any(g.abs().sum() > 0 for n,g in grads.items() if n.startswith(prefix)), prefix
        if step == 1:
            assert model.serial_project[0].weight.grad[256:].abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); assert torch.isfinite(norm)
        optimizer.step(); assert model.fixed_state_digest() == frozen
        steps.append(dict(step=step+1, loss=float(loss), norm=float(norm)))
    write_json(Path(a.output), dict(passed=True, no_AP=True, reference=reference, candidate=candidate,
        spatial_pack_inverse_exact=True, cyclic_projection_initially_injective=True,
        extra_hidden_channels_receive_gradient_after_first_update=True,
        shared_initial_fused_counts_equal=True, frames=frames, steps=steps,
        no_training_codec=True, discarded_updates=True, frozen_state_unchanged=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gradient_prefixes_checked=prefixes,
        MAC_scope='Dense Conv2d/ConvTranspose2d multiply-accumulates only; not total compute or hardware performance'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'source', 'output'): p.add_argument('--'+k, required=True)
    run(p.parse_args())

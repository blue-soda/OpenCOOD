"""Real-data admission for relocating the single paid message to conv3."""
import argparse
import copy
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec, run as quota_audit
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


def run(a):
    quota_audit(a)  # Explicit wire tests; its training steps prohibit the codec.
    seed()
    cfg = load_yaml(a.config)
    model = train_utils.create_model(cfg).cuda()
    initial = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(initial, strict=True)
    source = torch.load(a.source, map_location='cpu')
    base_cfg = copy.deepcopy(cfg)
    base_cfg['model']['core_method'] = 'e3dsnn_snapshot_highres'
    base = train_utils.create_model(base_cfg).cuda()
    base.load_state_dict(source, strict=True)
    model.eval(); base.eval()
    candidate = cfg['model']['core_method'] == 'e3dsnn_snapshot_shallow_message'
    val = build_dataset(cfg, train=False)
    parity = []
    for indices in ([0], [1], [0, 1]):
        batch = train_utils.to_device(val.collate_batch_train([val[i] for i in indices]), torch.device('cuda'))
        ego = dict(batch['ego'], disable_road=True)
        with no_codec(), torch.no_grad():
            out = model(ego); expected = base(ego)
        assert all(torch.equal(out[k], expected[k]) for k in ('psm', 'rm', 'dm'))
        if candidate:
            assert model._ego_deep is None and model._local_shallow is None and model._local_bev is None
        parity.append(dict(indices=indices, zero_message_source_bitwise=True))
    if candidate:
        captured = []
        h = model.backbone_3d.conv3.register_forward_hook(lambda m, i, o: captured.append(o))
        with torch.no_grad():
            E3dsnnSingle.encode_lidar(model, batch['ego']['road_lidar'], 2)
        h.remove()
        with torch.no_grad():
            prefix = model.encode_lidar(batch['ego']['road_lidar'], 2)
            assert torch.equal(prefix.indices, captured[0].indices)
            assert torch.equal(prefix.features, model.shallow_readout(captured[0].features))
        # Physical origin/step and direction, independent of the detection head.
        x = torch.zeros(1, 1, 5, 5, 5, device='cuda'); x[0, 0, 2, 2, 2] = 1
        pose = torch.eye(4, device='cuda').unsqueeze(0)
        for axis in range(3):
            t = pose.clone(); t[0, axis, 3] = model.step[axis]
            warped, _ = warp_volume(x, t, model.origin, model.step, geometry_dtype=torch.float64)
            target = torch.zeros_like(x); loc = [2, 2, 2]; loc[2-axis] += 1
            target[(0, 0) + tuple(loc)] = 1
            assert torch.allclose(warped, target, atol=1e-5, rtol=0)
        del captured, prefix
    del base, source
    model.train(); seed()
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
        assert len(observed) == (2 if candidate else 1)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in observed)
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        for prefix in ('content.project.', 'content.expand.', 'joint_receiver.', 'local_detail.',
                       'backbone_3d.conv3.', 'backbone_3d.conv4.', 'backbone_2d.'):
            assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith(prefix)), prefix
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); assert torch.isfinite(norm)
        optimizer.step(); assert model.fixed_state_digest() == frozen
        rows.append(dict(step=step+1, loss=float(loss), norm=float(norm),
                         shallow_input_gradient_l1=[float(x.grad.abs().sum()) for x in observed]))
    h.remove()
    write_json(Path(a.output).with_name('gradient_admission.json'), dict(passed=True,
        candidate=candidate, zero_message_parity=parity, steps=rows,
        road_prefix_equivalent=candidate, physical_xyz_translation_checked=candidate,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_state_unchanged=True, no_training_codec=True, updates_discarded=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'source', 'output'): p.add_argument('--'+k, required=True)
    run(p.parse_args())

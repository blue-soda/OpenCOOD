"""Frozen-weight admission: readout identity, IF equivalence and train-only probes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import write_json
from opencood.models.e3dsnn_snapshot_precision import FixedRangeCount
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike


def check_count_math():
    # Dyadic grid includes every threshold and saturation boundary exactly;
    # additional random values check the closed form away from boundaries.
    x = torch.cat((torch.arange(-256, 1281, device='cuda') / 256.,
                   torch.randn(10000, device='cuda') * 3)).requires_grad_()
    old = Multispike()(x)
    new = FixedRangeCount(4)(x)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old.sum(), x)[0],
                       torch.autograd.grad(new.sum(), x)[0])
    for steps in (4, 8, 16):
        theta = 4. / steps
        charge = x.detach().clamp(0, 4) / steps
        membrane = torch.full_like(charge, theta / 2)
        counts = torch.zeros_like(charge)
        for _ in range(steps):
            membrane = membrane + charge
            spike = (membrane >= theta).to(charge.dtype)
            counts += spike
            membrane -= spike * theta
        assert torch.equal(counts * theta, FixedRangeCount(steps)(x.detach()))


def run(a):
    seed(); torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35)
    check_count_math()
    cfg = load_yaml(a.config)
    model = train_utils.create_model(cfg).cuda().eval()
    ref = train_utils.create_model(load_yaml(a.reference_config)).cuda().eval()
    state = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True); ref.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in ref.parameters()) == 3374583
    # All modes must pass old Count4 identity before activating the intervention.
    readout = model.shallow_readout; model.shallow_readout = FixedRangeCount(4)
    val = build_dataset(cfg, train=False)
    for i in (0, 1):
        batch = train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda'))
        with no_codec(), torch.no_grad():
            got = model(batch['ego']); expected = ref(batch['ego'])
        assert all(torch.equal(got[k], expected[k]) for k in ('psm', 'rm', 'dm'))
        boxes, scores, gt = val.post_process_no_fusion(batch, {'ego': got})
        with np.load(str(Path(a.reference_predictions) / ('%06d.npz' % i)), allow_pickle=False) as z:
            assert np.array_equal(boxes.cpu().numpy(), z['pred_boxes'])
            assert np.array_equal(scores.cpu().numpy(), z['scores'])
            assert np.array_equal(gt.cpu().numpy(), z['gt_boxes'])
    model.shallow_readout = readout
    train = build_dataset(cfg, train=True)
    rows = []
    for i in range(32):
        batch = train_utils.to_device(train.collate_batch_test([train[i]]), torch.device('cuda'))
        captured = {}
        def observe(module, inputs, output):
            x = inputs[0].detach(); y = output.detach(); target = x.clamp(0, 4)
            assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 4
            if model.precision_mode != 'continuous':
                theta = 4. / module.steps
                assert torch.equal(y / theta, (y / theta).round())
            captured.update(mse=float((target-y).square().mean()),
                            count4_mse=float((target-Multispike()(x)).square().mean()),
                            changed_fraction=float((y != Multispike()(x)).float().mean()),
                            saturation_fraction=float((x >= 4).float().mean()))
        h = model.shallow_readout.register_forward_hook(observe)
        with no_codec(), torch.no_grad(): got = model(batch['ego']); expected = ref(batch['ego'])
        h.remove()
        assert not got['communication_measured'] and 'packet_bytes' not in got
        differences = {k: float((got[k]-expected[k]).abs().max()) for k in ('psm', 'rm', 'dm')}
        rows.append(dict(index=i, readout=captured, raw_max_difference=differences))
    assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
    changed = sum(any(v > 0 for v in r['raw_max_difference'].values()) for r in rows)
    assert changed == (0 if model.precision_mode == 'count4' else 32)
    write_json(Path(a.output), dict(passed=True, mode=model.precision_mode,
        source_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        parameters=3374583, unchanged_weights=True, count4_output_gradient_parity=True,
        count_if_equivalence=True, original_archived_two_frame_parity=True,
        no_codec=True, no_optimizer=True, no_ap_selection=True, changed_frames=changed, frames=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'reference-config', 'reference-predictions', 'output'):
        p.add_argument('--'+k, required=True)
    run(p.parse_args())

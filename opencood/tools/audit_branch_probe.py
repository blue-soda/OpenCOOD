"""Real-data engineering admission; no AP or branch-necessity claim."""
import argparse
import copy
import hashlib
from pathlib import Path
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.audit_snapshot_event_quota import no_codec
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(a):
    seed(); torch.set_num_threads(1)
    cfg = load_yaml(a.config)
    initial = torch.load(a.checkpoint, map_location='cpu')
    val = build_dataset(cfg, train=False)
    train = build_dataset(cfg, train=True)
    samples = [train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda')) for i in (0, 1)]
    train_batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
    reference = train_utils.create_model(cfg).cuda()
    reference.load_state_dict(initial, strict=True); reference.eval()
    with no_codec(), torch.no_grad():
        expected = [{k: v.detach().cpu() for k, v in reference(s['ego']).items() if k in ('psm', 'rm', 'dm')} for s in samples]
    del reference
    reports = []; common_fused = None
    for mode in ('dual', 'deep', 'shallow'):
        seed(); c = copy.deepcopy(cfg)
        c['model']['core_method'] = 'e3dsnn_snapshot_branch_probe'
        c['model']['args']['branch_mode'] = mode
        model = train_utils.create_model(c).cuda()
        removed = {'dual': (), 'deep': ('local_detail.',),
                   'shallow': ('backbone_3d.conv4.', 'backbone_3d.conv_out.', 'backbone_2d.')}[mode]
        state = {k: v for k, v in initial.items() if not k.startswith(removed)}
        model.load_state_dict(state, strict=True)
        assert all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items())
        assert not any(k.startswith(removed) for k in model.state_dict())
        model.eval(); fused = []
        original_detect = model.detect_bev_sample
        def capture(spatial, sample_index):
            fused.append(digest(spatial))
            return original_detect(spatial, sample_index)
        model.detect_bev_sample = capture
        frames = []
        for i, sample in enumerate(samples):
            with no_codec(), torch.no_grad():
                direct = model(sample['ego'])
            if mode == 'dual':
                assert all(torch.equal(direct[k].cpu(), expected[i][k]) for k in expected[i])
            assert not direct['communication_measured'] and 'packet_bytes' not in direct
            frames.append(dict(index=i, fused_sha256=fused[-1], events=direct['stream_events']))
        if common_fused is None:
            common_fused = fused[:]
        assert fused == common_fused, (mode, 'Different collaborative input')
        model.detect_bev_sample = original_detect
        with no_codec(), torch.no_grad():
            direct = model(samples[0]['ego'])
        model.packet_roundtrip = True
        with torch.no_grad():
            wire = model(samples[0]['ego'])
        assert all(torch.equal(direct[k], wire[k]) for k in ('psm', 'rm', 'dm'))
        assert wire['communication_measured'] and wire['packet_bytes'][0] > 0
        model.packet_roundtrip = False
        model.train(); model.message_dropout = 0.
        frozen = model.fixed_state_digest()
        optimizer = make_optimizer(c, model); criterion = train_utils.create_loss(c).cuda()
        prefixes = ['content.project.', 'content.expand.', 'joint_receiver.', 'backbone_3d.conv3.', 'cls_head.', 'reg_head.']
        if mode != 'deep': prefixes.append('local_detail.')
        if mode != 'shallow': prefixes += ['backbone_3d.conv4.', 'backbone_3d.conv_out.', 'backbone_2d.']
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with no_codec():
                output, loss = checked_forward(model, criterion, train_batch)
                loss.backward()
            assert not output['communication_measured'] and 'packet_bytes' not in output
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in grads.values())
            for prefix in prefixes:
                assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith(prefix)), (mode, prefix)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            assert torch.isfinite(norm)
            optimizer.step(); assert model.fixed_state_digest() == frozen
            steps.append(dict(step=step+1, loss=float(loss), norm=float(norm)))
        reports.append(dict(mode=mode, frames=frames, steps=steps,
            removed_prefixes=removed, exact_retained_initialization=True,
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            wire_bytes_first_frame=wire['packet_bytes'][0], wire_predictions_bitwise_equal=True,
            gradient_prefixes_checked=prefixes, frozen_state_unchanged=True))
        del model, optimizer, criterion, output, loss, grads, state
        torch.cuda.empty_cache()
    write_json(Path(a.output), dict(passed=True, arms=reports, no_AP=True,
        no_training_codec=True, discarded_optimization_steps_per_arm=2,
        dual_matches_original=True, identical_fused_input_two_frames=True,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'output'): p.add_argument('--'+k, required=True)
    run(p.parse_args())

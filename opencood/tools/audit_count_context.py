"""Admission and frozen counterfactual checks for native count-context study."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.audit_snapshot_event_quota import no_codec, run as quota_audit
from opencood.tools.audit_snapshot_native import seed
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.utils.cp_optimizer import make_optimizer

KEYS = ('psm', 'rm', 'dm')


def run(a):
    seed(); torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35)
    cfg = load_yaml(a.config)
    model = train_utils.create_model(cfg).cuda()
    initial = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(initial, strict=True); model.eval()
    report = dict(mode=model.context_mode, checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest())
    if a.mode == 'admit':
        ref = train_utils.create_model(load_yaml(a.reference_config)).cuda()
        ref.load_state_dict(torch.load(a.reference_checkpoint, map_location='cpu'), strict=True); ref.eval()
        val = build_dataset(cfg, train=False)
        for i in (0, 1):
            batch = train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda'))
            with no_codec(), torch.no_grad():
                got = model(batch['ego']); expected = ref(batch['ego'])
            assert all(torch.equal(got[k], expected[k]) for k in KEYS)
            # Tie the new checkout to archived terminal predictions, not just
            # another model instantiated from the same new implementation.
            boxes, scores, gt = val.post_process_no_fusion(batch, {'ego': got})
            with np.load(str(Path(a.reference_predictions) / ('%06d.npz' % i)), allow_pickle=False) as z:
                assert np.array_equal(boxes.cpu().numpy(), z['pred_boxes'])
                assert np.array_equal(scores.cpu().numpy(), z['scores'])
                assert np.array_equal(gt.cpu().numpy(), z['gt_boxes'])
        del ref
        report['initial_raw_and_archived_prediction_parity'] = True
        if model.count_context is not None:
            with torch.no_grad(): model.count_context.output.weight.normal_(0, .005)
            batch = train_utils.to_device(val.collate_batch_train([val[0], val[1]]), torch.device('cuda'))
            with no_codec(), torch.no_grad(): batched = model(batch['ego'])
            for i in (0, 1):
                one = train_utils.to_device(val.collate_batch_test([val[i]]), torch.device('cuda'))
                with no_codec(), torch.no_grad(): alone = model(one['ego'])
                for k in KEYS:
                    torch.testing.assert_close(batched[k][i:i+1], alone[k], rtol=1e-3, atol=1e-4)
            report['nonzero_projection_batch_routing'] = True
        train = build_dataset(cfg, train=True)
        batch = train_utils.to_device(train.collate_batch_train([train[0], train[1]]), torch.device('cuda'))
        # Actual cold-start updates: first step learns the zero output, second
        # must propagate to both local and context slices of the input conv.
        model.load_state_dict(initial, strict=True); model.train(); model.message_dropout = 0.
        optimizer = make_optimizer(cfg, model); criterion = train_utils.create_loss(cfg).cuda()
        frozen = model.fixed_state_digest(); history = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with no_codec():
                out, loss = checked_forward(model, criterion, batch); loss.backward()
            assert not out['communication_measured'] and 'packet_bytes' not in out
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in grads.values())
            context_grad = None
            if model.count_context is not None:
                assert grads['count_context.output.weight'].abs().sum() > 0
                if step == 1:
                    grad = grads['count_context.mix.weight']
                    assert grad[:, :64].abs().sum() > 0 and grad[:, 64:].abs().sum() > 0
                    context_grad = float(grad[:, 64:].abs().sum())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            assert torch.isfinite(norm); optimizer.step()
            assert model.fixed_state_digest() == frozen
            history.append(dict(step=step+1, loss=float(loss), gradient_norm=float(norm), context_input_gradient_l1=context_grad))
        if model.count_context is not None:
            assert all(not torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items() if k.startswith('count_context.'))
        report.update(discarded_optimizer_steps=history, frozen_state_unchanged=True)
        model.load_state_dict(initial, strict=True); model.eval()
        if model.count_context is not None:
            with torch.no_grad(): model.count_context.output.weight.normal_(0, .005)
        frames = 4
    else:
        frames = a.frames

    train = build_dataset(cfg, train=True)
    rows = []
    for i in range(frames):
        batch = train_utils.to_device(train.collate_batch_test([train[i]]), torch.device('cuda'))
        captured = {}
        handles = []
        if model.count_context is not None:
            def observe(module, inputs, output):
                captured.update(local=inputs[0].detach(), context=inputs[1].detach(), delta=output.detach())
            def readout(module, inputs, output): captured['counts'] = output.detach()
            handles = [model.count_context.register_forward_hook(observe), model.count_context.readout.register_forward_hook(readout)]
        with no_codec(), torch.no_grad(): normal = model(batch['ego'])
        for h in handles: h.remove()
        row = dict(index=i)
        if captured:
            for key in ('local', 'context', 'counts'):
                x = captured[key]
                assert torch.isfinite(x).all() and torch.equal(x, x.round())
            assert captured['local'].min() >= 0 and captured['local'].max() <= 4
            assert captured['counts'].min() >= 0 and captured['counts'].max() <= 4
            assert captured['context'].min() >= -4 and captured['context'].max() <= 4
            assert captured['delta'].abs().max() <= 1
            row.update(context_nonzero_fraction=float((captured['context'] != 0).float().mean()), delta_abs_mean=float(captured['delta'].abs().mean()))
            for mode in ('zero_context', 'shift_context', 'zero_branch'):
                model.context_intervention = mode
                with no_codec(), torch.no_grad(): changed = model(batch['ego'])
                row[mode] = {k: float((normal[k] - changed[k]).abs().max()) for k in KEYS}
            model.context_intervention = 'normal'
            assert model._context_local is None and model._context_road is None
        rows.append(row)
    if a.mode == 'admit' and model.count_context is not None:
        assert any(r['context_nonzero_fraction'] > 0 for r in rows)
        assert all(any(r[mode][k] > 0 for r in rows for k in KEYS) for mode in ('zero_context', 'shift_context', 'zero_branch'))
    if a.mode == 'diagnose':
        assert all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items())
    report.update(passed=True, frames=rows, no_codec=True, no_ap_selection=True,
                  diagnostic_projection_randomized=a.mode == 'admit' and model.count_context is not None,
                  terminal_zero_effect_is_valid=a.mode == 'diagnose')
    if a.mode == 'admit':
        del model; torch.cuda.empty_cache()
        # Includes direct-vs-wire bitwise parity, zero-message invariance and
        # two independently discarded train updates at the registered quota.
        quota_audit(a)
        write_json(Path(a.output).with_name('context_admission.json'), report)
    else:
        write_json(Path(a.output), report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'checkpoint', 'output'): p.add_argument('--'+k, required=True)
    for k in ('reference-config', 'reference-checkpoint', 'reference-predictions'): p.add_argument('--'+k)
    p.add_argument('--mode', choices=['admit', 'diagnose'], default='admit')
    p.add_argument('--frames', type=int, default=32)
    run(p.parse_args())

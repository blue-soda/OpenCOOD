"""Inference-only ECTRA bypass diagnostics on an unchanged checkpoint."""
import argparse
import sys
import types

import torch
from opencood.tools import inference, train_utils


def bypass_roi(self, coarse_grid, reserved_mask, *args, **kwargs):
    return coarse_grid, reserved_mask, {}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ablation', choices=['none', 'bypass_dense', 'bypass_roi'], required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    original_create = train_utils.create_model

    def create_model(config):
        model = original_create(config)
        if not hasattr(model, 'ectra_dense_enabled') or model.ectra_roi is None:
            raise ValueError('This diagnostic requires an ECTRA model with ROI refinement.')
        if args.ablation == 'bypass_dense':
            model.ectra_dense_enabled = False
        if args.ablation == 'bypass_roi':
            model.ectra_roi.forward = types.MethodType(bypass_roi, model.ectra_roi)
        calls = [0]

        def feature_stats(module, inputs, output):
            if calls[0] >= 5:
                return
            calls[0] += 1
            with torch.no_grad():
                before, after = inputs[0].detach(), output[0].detach()
                record_len = inputs[1]
                k = before.shape[0] // int(record_len.sum())
                offset = 0
                indices = []
                for n in record_len.tolist():
                    indices.extend(offset + j * k for j in range(1, n))
                    offset += n * k
                if not indices:
                    return
                before, after = before[indices], after[indices]
                support = (before.abs().amax(dim=1, keepdim=True) > 0)
                active_change = ((after - before).abs() * support).sum() / (
                    support.sum() * before.shape[1]).clamp_min(1)
                print('ECTRA_FEATURE_STATS', {
                    'time_intervals': inputs[2].detach().cpu().tolist()
                    if len(inputs) > 2 and torch.is_tensor(inputs[2]) else None,
                    'before_rms': before.square().mean().sqrt().item(),
                    'after_rms': after.square().mean().sqrt().item(),
                    'before_nonzero': (before != 0).float().mean().item(),
                    'after_nonzero': (after != 0).float().mean().item(),
                    'after_negative': (after < 0).float().mean().item(),
                    'relative_change': ((after-before).square().mean().sqrt()
                                        / before.square().mean().sqrt().clamp_min(1e-6)).item(),
                    'active_abs_change': active_change.item(),
                    'final_trust_mean': output[1]['ectra_trust'].mean().item()
                    if 'ectra_trust' in output[1] else None,
                })
        model.ectra.register_forward_hook(feature_stats)
        gate_calls = [0]

        def gate_stats(module, inputs, output):
            if gate_calls[0] >= 5:
                return
            gate_calls[0] += 1
            with torch.no_grad():
                gates = output.detach().sigmoid()
                print('ECTRA_GATE_STATS', {
                    'prediction_observation_update_means':
                        gates.mean(dim=(0, 2, 3)).cpu().tolist(),
                    'write_mean': (gates[:, 1] * gates[:, 2]).mean().item(),
                })

        model.ectra.trust_net.register_forward_hook(gate_stats)
        print('INFERENCE-ONLY ABLATION:', args.ablation,
              '(not a retrained ablation; weights and BN buffers are unchanged)')
        return model

    train_utils.create_model = create_model
    try:
        inference.main()
    finally:
        train_utils.create_model = original_create


if __name__ == '__main__':
    main()

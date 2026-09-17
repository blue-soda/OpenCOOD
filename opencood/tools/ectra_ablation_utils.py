"""Evaluation-only interventions on a fixed ECTRA checkpoint."""

import json
from pathlib import Path
from types import MethodType

import torch


ABLATIONS = ('none', 'recurrent_bypass', 'motion_identity', 'motion_no_decay',
             'calib_identity', 'trust_uniform', 'candidate_zero',
             'roi_context_zero', 'roi_refiner_bypass', 'roi_flow_bypass',
             'roi_trust_bypass')


def install_ablation(model, mode):
    if mode not in ABLATIONS:
        raise ValueError('Unknown ECTRA ablation: %s' % mode)
    if model.training:
        raise ValueError('Ablations are evaluation-only')
    if mode == 'none':
        return []
    recurrent = getattr(model, 'ectra', None)
    refiner = getattr(model, 'ectra_roi', None)
    if recurrent is None or refiner is None:
        raise ValueError('This ablation requires both ECTRA branches')
    handles = []
    if mode == 'recurrent_bypass':
        handles.append(recurrent.register_forward_hook(
            lambda module, inputs, output: (inputs[0], output[1])))
    elif mode == 'motion_identity':
        def propagate(self, hidden, ego_ref, dt, roi_context=None):
            return hidden, torch.zeros_like(hidden[:, :2]), torch.ones_like(hidden[:, :1])
        recurrent._propagate = MethodType(propagate, recurrent)
    elif mode == 'motion_no_decay':
        original = recurrent._propagate
        def propagate(self, hidden, ego_ref, dt, roi_context=None):
            _, flow, gamma = original(hidden, ego_ref, dt, roi_context)
            return self._warp_by_flow(hidden, flow), flow, torch.ones_like(gamma)
        recurrent._propagate = MethodType(propagate, recurrent)
    elif mode in ('calib_identity', 'trust_uniform', 'candidate_zero'):
        module = {'calib_identity': recurrent.calib_net,
                  'trust_uniform': recurrent.trust_net,
                  'candidate_zero': recurrent.candidate_net}[mode]
        handles.append(module.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)))
    elif mode == 'roi_context_zero':
        original = recurrent._format_roi_context
        def format_context(self, context, like):
            result = original(context, like)
            return torch.zeros_like(result) if result is not None else None
        recurrent._format_roi_context = MethodType(format_context, recurrent)
    elif mode.startswith('roi_'):
        def override(module, inputs, output):
            grid, mask, aux = output
            if mode in ('roi_refiner_bypass', 'roi_flow_bypass'):
                grid = inputs[0]
            if mode in ('roi_refiner_bypass', 'roi_trust_bypass'):
                mask = inputs[1]
            return grid, mask, aux
        handles.append(refiner.register_forward_hook(override))
    return handles


def install_diagnostics(model, path, mode):
    """Log scalar outputs and collaborator feature scale, without retaining graphs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.touch()
    latest = {}
    handles = []
    recurrent = getattr(model, 'ectra', None)
    if recurrent is None:
        raise ValueError('ECTRA diagnostics require an ECTRA model')

    def recurrent_hook(module, inputs, output):
        features, lengths = inputs[:2]
        frames = features.shape[0] // int(lengths.sum().item())
        start = 0
        before, after = [], []
        for count in lengths.tolist():
            for cav in range(1, count):
                index = start + cav * frames
                before.append(features[index].detach().abs().mean())
                after.append(output[0][index].detach().abs().mean())
            start += count * frames
        if before:
            latest['collaborator_input_abs'] = torch.stack(before).mean().item()
            latest['collaborator_output_abs'] = torch.stack(after).mean().item()
        for key, value in output[1].items():
            if torch.is_tensor(value) and value.numel() == 1:
                latest[key] = value.detach().item()

    def model_hook(module, inputs, output):
        row = dict(latest)
        latest.clear()
        index = inputs[0].get('sample_idx')
        if torch.is_tensor(index):
            index = index.detach().cpu().reshape(-1).tolist()
        elif hasattr(index, 'tolist'):
            index = index.tolist()
        row.update(sample_idx=index, ablation=mode)
        intervals = inputs[0].get('past_k_time_interval')
        if torch.is_tensor(intervals):
            intervals = intervals.detach().cpu().tolist()
        elif hasattr(intervals, 'tolist'):
            intervals = intervals.tolist()
        row['time_intervals'] = intervals
        for key, value in output.items():
            if key.startswith('ectra') and torch.is_tensor(value) and value.numel() == 1:
                row[key] = value.detach().item()
        with destination.open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    handles.append(recurrent.register_forward_hook(recurrent_hook))
    handles.append(model.register_forward_hook(model_hook))
    return handles

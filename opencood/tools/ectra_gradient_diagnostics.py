"""Opt-in local input sensitivity probes; never an attribution of AP gains."""

import torch


class EctraInputGradientProbe:
    def __init__(self, recurrent):
        c = recurrent.feature_dim
        self.layouts = {
            'motion_net': [('hidden', c), ('ego', c), ('time', 2)],
            'calib_net': [('observation', c), ('ego', c), ('hidden', c), ('time', 2)],
            'trust_net': [('hidden', c), ('observation', c), ('ego', c),
                          ('pred_residual', 1), ('obs_residual', 1), ('time', 2)],
            'candidate_net': [('hidden', c), ('observation', c), ('ego', c),
                              ('pred_trust', 1), ('obs_trust', 1)],
        }
        self.records = []
        self.handles = []
        for name in self.layouts:
            self.handles.append(getattr(recurrent, name).register_forward_pre_hook(
                self._capture(name)))

    def _capture(self, name):
        def hook(module, inputs):
            value = inputs[0]
            # Frozen encoder outputs otherwise have no input gradient. This
            # enables a leaf probe without unfreezing or detaching any encoder.
            if not value.requires_grad:
                value.requires_grad_(True)
            self.records.append((name, value))
        return hook

    def clear(self):
        self.records.clear()

    def summarize(self, loss):
        if not self.records:
            return []
        gradients = torch.autograd.grad(
            loss, [value for _, value in self.records],
            retain_graph=True, allow_unused=True)
        rows = []
        for (name, value), gradient in zip(self.records, gradients):
            row = {'module': name, 'connected': gradient is not None, 'branches': {}}
            offset = 0
            layout = list(self.layouts[name])
            remaining = value.shape[1] - sum(width for _, width in layout)
            if remaining < 0:
                raise ValueError('Unexpected ECTRA input channel layout')
            if remaining:
                layout.append(('roi_context', remaining))
            regions = {}
            if remaining:
                core_width = value.shape[1] - remaining
                roi = (value.detach()[:, core_width:core_width + 1] > 0).to(value.dtype)
                ego_offset = 0
                for branch, width in self.layouts[name]:
                    if branch == 'ego':
                        ego = value.detach()[:, ego_offset:ego_offset + width]
                        break
                    ego_offset += width
                regions = {'roi': roi,
                           'roi_ego_overlap': roi * (ego.abs().amax(dim=1, keepdim=True) > 0)}
                row['region_fraction'] = {k: mask.mean().item() for k, mask in regions.items()}
            for branch, width in layout:
                x = value.detach()[:, offset:offset + width]
                g = None if gradient is None else gradient.detach()[:, offset:offset + width]
                row['branches'][branch] = {
                    'activation_abs_mean': x.abs().mean().item(),
                    'gradient_abs_mean': None if g is None else g.abs().mean().item(),
                    'gradient_times_input_abs_mean': None if g is None else (g * x).abs().mean().item(),
                }
                for region, mask in regions.items():
                    denominator = mask.sum() * width
                    row['branches'][branch][region + '_gradient_times_input_abs_mean'] = (
                        None if g is None or denominator.item() == 0 else
                        ((g * x).abs() * mask).sum().div(denominator).item())
                offset += width
            rows.append(row)
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

"""Single-frame local membrane integration after the audited BEV warp/mask.

This is an isolated frozen-weights receiver probe, not an SNN encoder or a
temporal recurrent state. Count readout remains a separate downstream hook.
"""
import json
from pathlib import Path
import torch


def integrate(x, theta, mode, suppress_remote=False):
    if x.ndim != 4 or x.shape[0] != 2 or theta.shape != (1, x.shape[1], 1, 1):
        raise ValueError('Two already aligned/masked participants required')
    if not torch.isfinite(x).all() or (x < 0).any() or not torch.isfinite(theta).all() or (theta <= 0).any():
        raise ValueError('Nonnegative currents and positive thresholds required')
    local, road = x[:1], x[1:2]
    if mode == 'fixed':
        gate = torch.full_like(local, .5)
    elif mode == 'local':
        gate = theta / (theta + local)
    else:
        raise ValueError('Unsupported integration mode')
    current = gate * (torch.zeros_like(road) if suppress_remote else road)
    return local + current, gate, current


class CoSDHMembrane:
    def __init__(self, model, thresholds, mode, readout_steps, suppress_remote=False):
        assert mode in ('identity', 'fixed', 'local') and readout_steps in (4, 16)
        if suppress_remote and mode != 'local':
            raise ValueError('Remote-current ablation preserves the local gate')
        self.suppress_remote = suppress_remote
        values = json.loads(Path(thresholds).read_text())
        assert values['interface'] == 'post_fusion'
        self.mode = mode; self.handles = []; self.rows = {}; self.calls = [0, 0, 0]
        for stage, fusion in enumerate(model.fusion_net):
            assert type(fusion).__name__ == 'Where2comm' and type(fusion.fuse_modules).__name__ == 'MaxFusion'
            theta = torch.tensor(values['stages'][str(stage)]['range_p999'],
                device=next(model.parameters()).device).reshape(1, -1, 1, 1) / readout_steps
            self.handles.append(fusion.fuse_modules.register_forward_hook(self.hook(stage, theta)))

    def hook(self, stage, theta):
        def apply(module, inputs, output):
            x, record_len = inputs[:2]
            assert record_len.tolist() == [2] and x.shape[0] == 2
            # The enclosing Where2comm already applied BEV warp and target mask.
            # Verify this exact interface instead of silently applying a second warp.
            assert torch.equal(output, x.max(dim=0, keepdim=True).values)
            self.calls[stage] += 1
            if self.mode == 'identity':
                self.rows[stage] = dict(stage=stage, identity=True, membrane_minus_max_abs_mean=0.)
                return output
            membrane, gate, current = integrate(x, theta, self.mode, self.suppress_remote)
            assert torch.isfinite(membrane).all()
            if self.suppress_remote:
                assert torch.equal(membrane, x[:1]) and torch.count_nonzero(current) == 0
            self.rows[stage] = dict(stage=stage, gate_mean=float(gate.detach().mean()),
                current_abs_mean=float(current.detach().abs().mean()),
                membrane_minus_max_abs_mean=float((membrane-output).detach().abs().mean()))
            if self.suppress_remote:
                self.rows[stage].update(remote_current_suppressed=True, membrane_equals_local=True)
            return membrane
        return apply

    def begin(self):
        self.rows = {}; self.calls = [0, 0, 0]

    def finish(self):
        assert self.calls == [1, 1, 1] and len(self.rows) == 3
        return dict(mode=self.mode, calls=self.calls[:], stages=[self.rows[i] for i in range(3)],
            equation='U + g(U) * decoded_aligned_masked_road',
            gate='none' if self.mode == 'identity' else ('0.5' if self.mode == 'fixed' else 'theta/(theta+U)'),
            local_only_gate=True, temporal_state=False, added_parameters=0,
            original_max_output_used=self.mode == 'identity')

    def close(self):
        for handle in self.handles: handle.remove()


def self_test():
    x = torch.tensor([[[[0., 1., 3.]]], [[[4., 2., 0.]]]])
    theta = torch.ones(1, 1, 1, 1)
    m, g, _ = integrate(x, theta, 'local')
    assert torch.equal(g, torch.tensor([[[[1., .5, .25]]]]))
    assert torch.equal(m, torch.tensor([[[[4., 2., 3.]]]]))
    y = x.clone(); y[1] *= 3
    assert torch.equal(integrate(y, theta, 'local')[1], g), 'Gate must not inspect road features'
    y[1].zero_()
    for mode in ('fixed', 'local'):
        assert torch.equal(integrate(y, theta, mode)[0], x[:1]), 'Zero road current must preserve local membrane'
    assert torch.equal(integrate(x, theta, 'fixed')[0], x[:1]+.5*x[1:])
    for factor in (0., 1., 100.):
        changed = x.clone(); changed[1] *= factor
        m, g, current = integrate(changed, theta, 'local', suppress_remote=True)
        assert torch.equal(m, x[:1]) and torch.count_nonzero(current) == 0
    z = x.clone().requires_grad_(True)
    integrate(z, theta, 'local', suppress_remote=True)[0].sum().backward()
    assert torch.equal(z.grad[0], torch.ones_like(z.grad[0])) and z.grad[1].count_nonzero() == 0
    z = x.clone().requires_grad_(True)
    integrate(z, theta, 'local')[0].sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad[1].count_nonzero() == 3
    assert torch.equal(x, torch.tensor([[[[0., 1., 3.]]], [[[4., 2., 0.]]]]))
    print('Membrane routing, local-only gate, zero-road identity and finite gradients passed')


if __name__ == '__main__': self_test()

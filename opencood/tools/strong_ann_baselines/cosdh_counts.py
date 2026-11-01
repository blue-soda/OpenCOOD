"""Frozen sender-only analytic IF count representation, without a wire codec."""
import json
from pathlib import Path
import torch


class _RemoteSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized, scale):
        ctx.save_for_backward(x, scale)
        return quantized

    @staticmethod
    def backward(ctx, grad):
        x, scale = ctx.saved_tensors
        return grad * ((x >= 0) & (x <= scale)).to(grad.dtype), None, None


def prefix_counts(rate, counts, quota):
    """GPU-only IF crossing-time prefix; ties use channel-major address."""
    if quota is None or quota >= int(counts.sum()):
        return counts
    if quota < 0:
        raise ValueError('Nonnegative event quota required')
    if quota == 0:
        return torch.zeros_like(counts)
    flat = counts.reshape(-1).long()
    k = torch.arange(1, int(flat.max())+1, device=rate.device)
    valid = k[None, :] <= flat[:, None]
    # Flattening neuron first fixes address order before the stable time sort.
    times = (k[None, :] / rate.reshape(-1, 1).clamp_min(1e-20))[valid]
    address = torch.arange(flat.numel(), device=rate.device)[:, None].expand_as(valid)[valid]
    selected = torch.sort(times, stable=True).indices[:quota]
    return torch.bincount(address[selected], minlength=flat.numel()).reshape_as(counts).to(counts.dtype)


def quantize_remote(x, scale, steps, quota=None, surrogate=False):
    if surrogate and quota is not None:
        raise ValueError('Quota selection has no registered surrogate')
    if x.ndim != 4 or x.shape[0] != 2 or scale.shape != (x.shape[1], 1, 1):
        raise ValueError('Expected two agents and a per-channel scale')
    if not torch.isfinite(x).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid input or calibration scale')
    if (x[1] < 0).any():
        raise ValueError('Unsigned IF input must be nonnegative')
    theta = scale / steps
    rate = x[1] / theta
    counts = prefix_counts(rate, torch.floor(rate).clamp(0, steps), quota)
    result = x.clone()
    value = counts * theta
    result[1] = _RemoteSurrogate.apply(x[1], value, scale) if surrogate else value
    assert torch.equal(result[0], x[0])
    return result, counts


def replace_remote(x, scale, steps):
    return quantize_remote(x, scale, steps)[0]


class CoSDHCounts:
    def __init__(self, model, thresholds, steps, quotas=None, surrogate=False):
        assert steps in (8, 32) and len(model.naive_compressor_list) == 3
        if quotas is not None and (len(quotas) != 3 or any(v < 0 for v in quotas)):
            raise ValueError('Three nonnegative stage quotas required')
        self.steps = steps; self.quotas = quotas; self.surrogate = surrogate; self.calls = [0, 0, 0]; self.handles = []
        self.current = {}; self.scales = {}
        values = json.loads(Path(thresholds).read_text())
        assert len(values['fit_ids']) == 64 and set(values['stages']) == {'0', '1', '2'}
        for stage, compressor in enumerate(model.naive_compressor_list):
            scale = torch.tensor(values['stages'][str(stage)]['range_p999'],
                                 device=next(model.parameters()).device).reshape(-1, 1, 1)
            self.scales[stage] = scale
            self.handles.append(compressor.decoder.register_forward_pre_hook(self.hook(stage, scale)))

    def hook(self, stage, scale):
        def apply(module, inputs):
            self.calls[stage] += 1
            result, counts = quantize_remote(inputs[0], scale, self.steps,
                None if self.quotas is None else self.quotas[stage], self.surrogate)
            self.current[stage] = counts.detach()
            return (result,) + inputs[1:]
        return apply

    def begin(self):
        self.calls = [0, 0, 0]
        self.current = {}

    def finish(self):
        assert self.calls == [1, 1, 1], self.calls
        return dict(steps=self.steps, decoder_calls=self.calls[:], remote_only=True,
                    codec_executed=False, event_selection=self.quotas is not None,
                    stage_quotas=self.quotas)

    def close(self):
        for handle in self.handles: handle.remove()


def self_test():
    rate = torch.tensor([2., 2., 4., .5])
    full = rate.floor()
    # Times .25(addr2), .5(addr0,addr1,addr2), .75(addr2), 1(addr0,addr1,addr2).
    expected = [2, 0, 1, 2, 2, 0, 1, 2]
    for quota in range(10):
        want = torch.bincount(torch.tensor(expected[:quota], dtype=torch.long), minlength=4).float()
        assert torch.equal(prefix_counts(rate, full, quota), want)
    assert torch.equal(prefix_counts(rate, full, None), full)
    for steps in (8, 32):
        x = torch.tensor([[[[-2., .1, 1., 2.]]], [[[0., .25, 1., 2.]]]])
        y = replace_remote(x, torch.ones(1, 1, 1), steps)
        assert torch.equal(x[0], y[0])
        assert torch.equal(y[1], torch.tensor([[[0., .25, 1., 1.]]]))
        assert torch.equal(x[1], torch.tensor([[[0., .25, 1., 2.]]]))
        z = x.clone().requires_grad_(True)
        train_value, _ = quantize_remote(z, torch.ones(1, 1, 1), steps, surrogate=True)
        assert torch.equal(train_value.detach(), y)
        train_value.sum().backward()
        assert torch.equal(z.grad[0], torch.ones_like(z[0]))
        assert torch.equal(z.grad[1], (x[1] <= 1).float())
    print('Remote-only count, saturation, exact grid, ego and input preservation passed')


if __name__ == '__main__':
    self_test()

"""Frozen sender-only analytic IF count representation, without a wire codec."""
import json
from pathlib import Path
import torch


def replace_remote(x, scale, steps):
    if x.ndim != 4 or x.shape[0] != 2 or scale.shape != (x.shape[1], 1, 1):
        raise ValueError('Expected two agents and a per-channel scale')
    if not torch.isfinite(x).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid input or calibration scale')
    if (x[1] < 0).any():
        raise ValueError('Unsigned IF input must be nonnegative')
    theta = scale / steps
    counts = torch.floor(x[1] / theta).clamp(0, steps)
    result = x.clone()
    result[1] = counts * theta
    assert torch.equal(result[0], x[0])
    return result


class CoSDHCounts:
    def __init__(self, model, thresholds, steps):
        assert steps in (8, 32) and len(model.naive_compressor_list) == 3
        self.steps = steps; self.calls = [0, 0, 0]; self.handles = []
        values = json.loads(Path(thresholds).read_text())
        assert len(values['fit_ids']) == 64 and set(values['stages']) == {'0', '1', '2'}
        for stage, compressor in enumerate(model.naive_compressor_list):
            scale = torch.tensor(values['stages'][str(stage)]['range_p999'],
                                 device=next(model.parameters()).device).reshape(-1, 1, 1)
            self.handles.append(compressor.decoder.register_forward_pre_hook(self.hook(stage, scale)))

    def hook(self, stage, scale):
        def apply(module, inputs):
            self.calls[stage] += 1
            return (replace_remote(inputs[0], scale, self.steps),) + inputs[1:]
        return apply

    def begin(self):
        self.calls = [0, 0, 0]

    def finish(self):
        assert self.calls == [1, 1, 1], self.calls
        return dict(steps=self.steps, decoder_calls=self.calls[:], remote_only=True,
                    codec_executed=False, event_selection=False)

    def close(self):
        for handle in self.handles: handle.remove()


def self_test():
    for steps in (8, 32):
        x = torch.tensor([[[[-2., .1, 1., 2.]]], [[[0., .25, 1., 2.]]]])
        y = replace_remote(x, torch.ones(1, 1, 1), steps)
        assert torch.equal(x[0], y[0])
        assert torch.equal(y[1], torch.tensor([[[0., .25, 1., 1.]]]))
        assert torch.equal(x[1], torch.tensor([[[0., .25, 1., 2.]]]))
    print('Remote-only count, saturation, exact grid, ego and input preservation passed')


if __name__ == '__main__':
    self_test()

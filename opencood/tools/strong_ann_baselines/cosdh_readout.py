"""Calibrated post-fusion readout; does not replace the ANN fusion operator.

An isolated migration probe before introducing a membrane/gate receiver.
All thresholds are fitted on training frames. No packets or state updates.
"""
import json
from pathlib import Path
import torch


def readout(x, scale, mode):
    if x.ndim != 4 or x.shape[0] != 1 or scale.shape != (1, x.shape[1], 1, 1):
        raise ValueError('Expected one fused ego and per-channel scales')
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError('Unsigned readout requires nonnegative finite features')
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Positive finite scales required')
    if mode == 'identity':
        return x
    if mode == 'clip':
        return torch.minimum(x, scale)
    if mode not in ('count4', 'count16'):
        raise ValueError('Unsupported readout')
    steps = int(mode[5:])
    theta = scale / steps
    return torch.floor(x / theta).clamp(0, steps) * theta


class CoSDHReadout:
    def __init__(self, model, thresholds, mode):
        values = json.loads(Path(thresholds).read_text())
        assert values['interface'] == 'post_fusion' and len(values['fit_ids']) == 64
        self.mode = mode; self.handles = []; self.calls = [0, 0, 0]
        for stage, fusion in enumerate(model.fusion_net):
            scale = torch.tensor(values['stages'][str(stage)]['range_p999'],
                device=next(model.parameters()).device).reshape(1, -1, 1, 1)
            self.handles.append(fusion.register_forward_hook(self.hook(stage, scale)))

    def hook(self, stage, scale):
        def apply(module, inputs, output):
            self.calls[stage] += 1
            return (readout(output[0], scale, self.mode),) + output[1:]
        return apply

    def begin(self):
        self.calls = [0, 0, 0]

    def finish(self):
        assert self.calls == [1, 1, 1], self.calls
        return dict(mode=self.mode, calls=self.calls[:], interface='post_fusion',
                    original_fusion_preserved=True, membrane_conversion=False)

    def close(self):
        for handle in self.handles: handle.remove()


def self_test():
    x = torch.tensor([[[[0., .24, .25, .99, 1., 2.]]]])
    scale = torch.ones(1, 1, 1, 1)
    assert readout(x, scale, 'identity') is x
    assert torch.equal(readout(x, scale, 'clip'), x.clamp(max=1))
    assert torch.equal(readout(x, scale, 'count4'), torch.tensor([[[[0., 0., .25, .75, 1., 1.]]]]))
    assert torch.equal(readout(x, scale, 'count16'), torch.floor(x*16).clamp(0,16)/16)
    assert x.max() == 2
    for bad in (-x, x * float('nan')):
        try: readout(bad, scale, 'count4')
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
    print('Readout identity, clipping, Count4/16, saturation and input preservation passed')


if __name__ == '__main__':
    self_test()

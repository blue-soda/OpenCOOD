"""Bounded learned synaptic gain on already-paid, aligned road current."""
import json
from pathlib import Path
import torch
from torch import nn


class CoSDHSynapse(nn.Module):
    def __init__(self, mode, channels):
        super().__init__()
        assert mode in ('stage', 'channel') and list(channels) == [64, 128, 256]
        self.mode = mode
        self.raw = nn.ParameterList([nn.Parameter(torch.zeros(1 if mode == 'stage' else c)) for c in channels])
        self.force_identity = False

    def forward(self, stage):
        value = self.raw[stage]
        gain = torch.ones_like(value) if self.force_identity else 2 * torch.sigmoid(value)
        return gain.reshape(1, -1, 1, 1)

    def statistics(self):
        return dict(mode=self.mode, identity_override=self.force_identity,
                    stages=[dict(mean=float(self(i).detach().mean()),
                                 min=float(self(i).detach().min()),
                                 max=float(self(i).detach().max())) for i in range(3)])


def install(model, mode, thresholds):
    if mode == 'none':
        return
    assert not hasattr(model, 'snn_synapse')
    values = json.loads(Path(thresholds).read_text())
    assert values['interface'] == 'post_fusion'
    model.add_module('snn_synapse', CoSDHSynapse(mode, [len(values['stages'][str(i)]['range_p999']) for i in range(3)]))


def self_test():
    from cosdh_membrane import integrate
    for mode in ('stage', 'channel'):
        synapse = CoSDHSynapse(mode, [64, 128, 256])
        assert sum(p.numel() for p in synapse.parameters()) == (3 if mode == 'stage' else 448)
        for stage, c in enumerate((64, 128, 256)):
            x = torch.ones(2, c, 2, 2); theta = torch.ones(1, c, 1, 1)
            original = integrate(x, theta, 'local')
            learned = integrate(x, theta, 'local', gain=synapse(stage))
            assert all(torch.equal(a, b) for a, b in zip(original, learned))
            learned[0].sum().backward()
            assert torch.isfinite(synapse.raw[stage].grad).all() and (synapse.raw[stage].grad > 0).all()
            with torch.no_grad(): synapse.raw[stage].fill_(1.)
            assert (synapse(stage) > 1).all() and (synapse(stage) < 2).all()
            y = x.clone(); y[1].zero_()
            assert torch.equal(integrate(y, theta, 'local', gain=synapse(stage))[0], x[:1])
            assert torch.equal(integrate(x, theta, 'local', suppress_remote=True, gain=synapse(stage))[0], x[:1])
        synapse.force_identity = True
        assert all(torch.equal(synapse(i), torch.ones_like(synapse(i))) for i in range(3))
    print('Synaptic initialization identity, bounds, gradients and zero-road identity passed')


if __name__ == '__main__': self_test()

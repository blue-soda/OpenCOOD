"""Matched receive-only probes; unsupported sites retain the original Count4."""
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import count_readout


def receive(membrane, current, supported, mode):
    local = count_readout(membrane)
    if mode == 'membrane':
        fused = count_readout(membrane + current)
    elif mode == 'prequantized':
        fused = count_readout(local + current)
    elif mode == 'continuous':
        fused = (membrane + current).clamp(0, 4)
    else:
        raise ValueError('Unknown registered receiver probe')
    return torch.where(supported, fused, local)


def self_test():
    u = torch.tensor([.49, .49, -.1, 4.2], requires_grad=True)
    r = torch.tensor([.04, .04, .6, .5], requires_grad=True)
    mask = torch.tensor([True, False, True, True])
    full = receive(u, r, mask, 'membrane')
    pre = receive(u, r, mask, 'prequantized')
    continuous = receive(u, r, mask, 'continuous')
    assert torch.equal(full, torch.where(mask, count_readout(u+r), count_readout(u)))
    assert full[0] == 1 and pre[0] == 0
    assert continuous[0] == (u+r)[0]
    assert full[1] == pre[1] == continuous[1] == 0
    assert full[-1] == pre[-1] == continuous[-1] == 4
    for mode in ('membrane', 'prequantized', 'continuous'):
        a = u.detach().clone().requires_grad_(True)
        b = r.detach().clone().requires_grad_(True)
        receive(a, b, mask, mode).sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        assert b.grad[0] == 1 and b.grad[1] == b.grad[-1] == 0
    print('Receiver probes: retained fraction, support, range and gradients passed')


if __name__ == '__main__':
    self_test()

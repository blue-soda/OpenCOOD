import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection


class SignedProjectionTests(unittest.TestCase):
    @torch.no_grad()
    def check_stream(self, device):
        block = nn.Sequential(nn.ConvTranspose2d(2, 2, 2, stride=2),
                              nn.BatchNorm2d(2), nn.ReLU()).to(device).eval()
        block[0].weight.copy_(torch.arange(-8, 8, device=device).reshape_as(block[0].weight))
        block[0].bias.fill_(.3)
        block[1].running_mean.fill_(.7)
        q = torch.tensor([.25, .5], device=device)
        arms = [DeepSpikeProjection(block, q, name) for name in ('dense', 'amplitude', 'unit')]
        pose = np.eye(4)
        bounds = (0, 0, -1, 4, 2, 1)
        deep = torch.zeros(1, 2, 2, 4, device=device)
        states = [p.fresh(deep)[1] for p in arms]
        change = torch.zeros_like(deep)
        change[0, :, 0, 2] = q * torch.tensor([3., -2.], device=device)
        for offset, value in [(0, deep + change), (0, deep), (.5, deep), (1, deep + change)]:
            moved = pose.copy()
            moved[0, 3] = offset
            written = torch.ones_like(value[:, :1], dtype=torch.bool)
            _, _, route = arms[0].align_task(value, pose, moved, bounds)
            updated = [p.advance(value, written, s, pose, moved, bounds, route)
                       for p, s in zip(arms, states)]
            states = [s for _, s in updated]
            ref = arms[0]
            w = ref.weight.reshape_as(block[0].weight).double()
            expected = F.conv_transpose2d(ref.levels(value).double(), w, stride=2).float()
            expected = block[2](block[1](expected * ref.scale + block[0].bias.view(1, -1, 1, 1)))
            for image, state in updated:
                self.assertTrue(torch.equal(states[0], state))
                self.assertTrue(torch.equal(image, expected))
            pose = moved
        # A subthreshold residual survives in the task state while send levels stay zero.
        first = .375 * q.view(1, 2, 1, 1)
        self.assertFalse(bool(ref.levels(first).any()))
        self.assertTrue(bool((ref.levels(first + first) == 1).all()))
        for p in arms:
            self.assertTrue(torch.equal(p.fresh(deep)[1], torch.zeros_like(states[0])))

    def test_cpu(self):
        self.check_stream('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_stream('cuda')

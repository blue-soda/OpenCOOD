"""Parameter budget, event semantics, geometry and temporal credit assignment."""
import unittest

import numpy as np
import torch
from torch import nn

from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory, make_fusion_config
from opencood.models.sub_modules.int_spiking_gru import (
    INTSpikingGRUFusion, INTSpikingGRUMemory, signed_event)


def meta(stamp, segment='a', x=0.):
    pose = np.eye(4); pose[0, 3] = x
    return dict(timestamp_us=stamp, segment=segment, pose=pose.tolist())


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__(); self.value = value

    def forward(self, x):
        return torch.full_like(x[:, :1], self.value)


class SpikingGRUTests(unittest.TestCase):
    def test_exact_parameter_layout_initialization_and_cache_budget(self):
        region = [0, 0, -1, 8, 4, 1]
        torch.manual_seed(303)
        ann = INTFeatureMemory('gru', 64, region).eval()
        torch.manual_seed(303)
        snn = INTSpikingGRUMemory(64, region).eval()
        self.assertEqual(sum(p.numel() for p in snn.parameters()), 259712)
        self.assertEqual(dict(ann.named_parameters()).keys(), dict(snn.named_parameters()).keys())
        self.assertEqual(ann.state_dict().keys(), snn.state_dict().keys())
        for name, value in ann.state_dict().items():
            self.assertTrue(torch.equal(value, snn.state_dict()[name]), name)
        layers = lambda m: [(name, type(layer)) for name, layer in m.named_modules()
                            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Sigmoid, nn.Tanh, nn.ReLU))]
        self.assertEqual(layers(ann), layers(snn))
        for module in (ann, snn):
            out, state, info = module.step(torch.rand(1, 64, 4, 8), None, meta(1000000))
            self.assertEqual(tuple(out.shape), (1, 64, 4, 8))
            self.assertEqual(tuple(state['value'].shape), (1, 65, 4, 8))
            self.assertEqual(info['state_bytes'], 65*4*8*4)

    def test_two_polarities_and_surrogate(self):
        voltage = torch.tensor([-.5, -.25, -.1, 0., .1, .25, .5], requires_grad=True)
        events = signed_event(voltage, .25)
        self.assertEqual(events.tolist(), [-1., -1., 0., 0., 0., 1., 1.])
        events.sum().backward()
        self.assertTrue(bool(torch.isfinite(voltage.grad).all()))
        self.assertTrue(bool((voltage.grad > 0).all()))

    def test_accumulation_reset_and_no_membrane_readout(self):
        for polarity in (-1., 1.):
            fusion = INTSpikingGRUFusion(make_fusion_config('gru', 1)).eval()
            fusion.reset_gate_conv = Constant(.5)
            fusion.update_gate_conv = Constant(.5)
            fusion.candidate_conv = Constant(polarity*.4)
            fusion.post_fusion_conv = nn.Identity()  # Expose only the original readout inputs.
            x, past = torch.ones(1, 1, 2, 3), torch.zeros(1, 2, 2, 3)
            first, state = fusion(x, [past], detach_state=False)
            self.assertEqual(float(first[:, :1].abs().sum()), 0.)
            self.assertTrue(torch.allclose(state[:, :1], torch.full_like(x, polarity*.2)))
            previous = state.clone()
            second, state = fusion(x, [state], delta_t=.1, detach_state=False)
            self.assertTrue(torch.allclose(second[:, :1], torch.full_like(x, polarity*.25)))
            self.assertTrue(torch.allclose(state[:, :1], torch.full_like(x, polarity*.05), atol=1e-7))
            self.assertTrue(torch.equal(previous[:, :1], torch.full_like(x, polarity*.2)))
            self.assertTrue(torch.allclose(state[:, -1], torch.full_like(state[:, -1], .1)))

    def test_future_readout_gradient_and_detach_boundary(self):
        for detach in (False, True):
            torch.manual_seed(7)
            module = INTSpikingGRUMemory(4, [0, 0, -1, 8, 4, 1]).eval()
            earlier = torch.randn(1, 4, 4, 8, requires_grad=True)
            _, state, _ = module.step(earlier, None, meta(1000000), detach_state=detach)
            output, _, _ = module.step(torch.randn_like(earlier), state, meta(1100000), detach_state=False)
            output.square().sum().backward()
            if detach:
                self.assertIsNone(earlier.grad)
            else:
                self.assertTrue(bool(torch.isfinite(earlier.grad).all()))
                self.assertGreater(float(earlier.grad.abs().sum()), 0.)
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)

    def test_alignment_reset_and_state_ownership(self):
        module = INTSpikingGRUMemory(1, [0, 0, -1, 4, 2, 1]).eval()
        module.fusion.reset_gate_conv = Constant(.5)
        module.fusion.update_gate_conv = Constant(.5)
        module.fusion.candidate_conv = Constant(0.)
        x = torch.zeros(1, 1, 2, 4)
        _, state, _ = module.step(x, None, meta(1000000))
        state['value'][0, :, 0, 2] = torch.tensor([.2, .1])
        original = state['value'].clone()
        _, shifted, info = module.step(x, state, meta(1100000, x=1.))
        self.assertTrue(info['warp'])
        self.assertTrue(torch.equal(state['value'], original))
        self.assertAlmostEqual(float(shifted['value'][0, 0, 0, 1]), .1, places=6)
        self.assertAlmostEqual(float(shifted['value'][0, 1, 0, 1]), .2, places=6)
        self.assertEqual(float(shifted['value'][:, :, :, -1].abs().sum()), 0.)
        for stamp, segment in ((1100000, 'b'), (1400000, 'a')):
            output, cleared, _ = module.step(x, state, meta(stamp, segment))
            fresh, expected, _ = module.step(x, None, meta(stamp, segment))
            self.assertTrue(torch.equal(output, fresh))
            self.assertTrue(torch.equal(cleared['value'], expected['value']))
        with self.assertRaises(ValueError):
            module.step(x, state, meta(1000000))

    def test_invalid_fixed_threshold(self):
        for value in (0., 1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                INTSpikingGRUMemory(4, [0, 0, -1, 8, 4, 1], threshold=value)


if __name__ == '__main__':
    unittest.main()

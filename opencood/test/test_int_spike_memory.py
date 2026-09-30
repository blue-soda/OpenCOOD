"""Temporal semantics: accumulation, binary events, alignment and credit assignment."""
import math
import unittest

import numpy as np
import torch
from torch import nn

from opencood.data_utils.int_tbptt import WindowOptimizer, window_ends
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory
from opencood.models.sub_modules.int_spike_memory import INTSpikeMemory, SurrogateSpike


def meta(stamp, segment='a', x=0.):
    pose = np.eye(4)
    pose[0, 3] = x
    return dict(timestamp_us=stamp, segment=segment, pose=pose.tolist(), reset_before=False)


def memory(mode='lif'):
    return INTSpikeMemory(mode, 1, [0, 0, -1, 4, 2, 1], memory_channels=1,
                          input_gain=1., tau_u=.2, tau_r=.1)


class SpikeSemantics(unittest.TestCase):
    def test_binary_forward_and_surrogate_backward(self):
        x = torch.tensor([-1., -.1, 0., .2, 1.], requires_grad=True)
        y = SurrogateSpike.apply(x)
        self.assertEqual(y.tolist(), [0., 0., 1., 1., 1.])
        y.sum().backward()
        self.assertTrue(bool((x.grad > 0).all()))

    def test_subthreshold_evidence_accumulates_then_soft_resets(self):
        module = memory()
        x = torch.full((1, 1, 2, 4), .7)
        first, state, _ = module.step(x, None, meta(1000000))
        self.assertEqual(float(first.sum()), 0.)
        second, state, _ = module.step(x, state, meta(1100000))
        self.assertTrue(torch.equal(second, torch.ones_like(second)))
        expected = .7 * math.exp(-.1/.2) + .7 - 1.
        self.assertTrue(torch.allclose(state['value'][:, :1], torch.full_like(x, expected), atol=1e-6))

    def test_no_spike_still_has_decaying_readout(self):
        module = memory()
        x = torch.full((1, 1, 2, 4), 1.1)
        _, state, _ = module.step(x, None, meta(1000000))
        out, _, info = module.step(torch.zeros_like(x), state, meta(1100000))
        self.assertEqual(info['emission_mean'], 0.)
        self.assertTrue(torch.allclose(out, torch.full_like(out, math.exp(-1)), atol=1e-6))

    def test_continuous_control_has_no_threshold_or_reset(self):
        module = memory('leaky')
        x = torch.full((1, 1, 2, 4), .7)
        first, state, _ = module.step(x, None, meta(1000000))
        self.assertTrue(torch.allclose(first, x, atol=1e-6))
        second, state, _ = module.step(x, state, meta(1100000))
        u = .7 * math.exp(-.5) + .7
        self.assertTrue(torch.allclose(state['value'][:, :1], torch.full_like(x, u), atol=1e-6))
        self.assertTrue(torch.allclose(second, torch.full_like(x, u+.7*math.exp(-1)), atol=1e-6))

    def test_both_state_channels_warp_and_new_regions_clear(self):
        module = memory()
        x = torch.zeros(1, 1, 2, 4)
        _, state, _ = module.step(x, None, meta(1000000))
        state['value'][0, :, 0, 2] = torch.tensor([.2, .8])
        before = state['value'].clone()
        _, after, info = module.step(x, state, meta(1100000, x=1.))
        self.assertTrue(torch.equal(state['value'], before))
        self.assertAlmostEqual(float(after['value'][0, 0, 0, 1]), .2*math.exp(-.5), places=6)
        self.assertAlmostEqual(float(after['value'][0, 1, 0, 1]), .8*math.exp(-1), places=6)
        self.assertEqual(float(after['value'][:, :, :, -1].abs().sum()), 0.)
        self.assertTrue(info['warp'])

    def test_reset_causality_budget_and_no_input_mutation(self):
        module = memory()
        x = torch.ones(1, 1, 2, 4)
        _, state, info = module.step(x, None, meta(1000000))
        self.assertEqual(info['state_bytes'], 2*x.numel()*4)
        self.assertFalse(state['value'].requires_grad)
        for next_meta, reason in [(meta(1100000, 'b'), 'scene'), (meta(1400000), 'gap')]:
            out, _, info = module.step(x, state, next_meta)
            fresh, _, _ = module.step(x, None, next_meta)
            self.assertTrue(torch.equal(out, fresh))
            self.assertEqual(info['reset'], reason)
        with self.assertRaises(ValueError):
            module.step(x, state, meta(1000000))

    def test_future_loss_reaches_unlabelled_input_but_not_across_detach(self):
        for detach in (False, True):
            module = memory()
            context = torch.full((1, 1, 2, 4), .7, requires_grad=True)
            _, state, _ = module.step(context, None, meta(1000000), detach_state=detach)
            output, _, _ = module.step(torch.full_like(context, .7), state, meta(1100000), detach_state=False)
            output.sum().backward()
            if detach:
                self.assertIsNone(context.grad)
            else:
                self.assertGreater(float(context.grad.abs().sum()), 0.)
                self.assertGreater(float(module.raw_tau_u.grad.abs().sum()), 0.)

    def test_ann_controls_can_also_backpropagate_across_frames(self):
        for mode in ('concat', 'gru'):
            torch.manual_seed(17)
            module = INTFeatureMemory(mode, 2, [0, 0, -1, 4, 2, 1], identity_init=False).eval()
            x = torch.ones(1, 2, 2, 4, requires_grad=True)
            _, state, _ = module.step(x, None, meta(1000000), detach_state=False)
            out, _, _ = module.step(torch.ones_like(x), state, meta(1100000), detach_state=False)
            out.square().sum().backward()
            self.assertGreater(float(x.grad.abs().sum()), 0., mode)


class TBPTTSemantics(unittest.TestCase):
    def test_windows_preserve_segments_and_include_trailing_context(self):
        rows = [meta(1000000+i*100000) for i in range(6)]
        rows += [meta(2000000+i*100000, 'b') for i in range(3)]
        self.assertEqual(window_ends(rows, 4), {3, 5, 8})
        rows[2]['reset_before'] = True
        self.assertEqual(window_ends(rows, 4), {1, 5, 8})

    def test_no_early_update_and_detach_is_not_reset(self):
        model = nn.Module()
        model.feature_memory = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.feature_memory.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        window = WindowOptimizer(model, optimizer)
        context = torch.ones(1, 1, requires_grad=True)
        state = dict(value=model.feature_memory(context))  # no label here
        state['value'] = state['value'] + model.feature_memory(torch.ones(1, 1))
        window.add(state['value'].square().mean())  # future labelled frame
        self.assertEqual(float(model.feature_memory.weight), 1.)
        retained = state['value'].detach().clone()
        state, stats = window.finish(state)
        self.assertTrue(stats['optimizer_step'])
        self.assertGreater(float(context.grad), 0.)
        self.assertTrue(torch.equal(state['value'], retained))
        self.assertFalse(state['value'].requires_grad)
        self.assertNotEqual(float(model.feature_memory.weight), 1.)
        before = model.feature_memory.weight.clone()
        state, stats = window.finish(state)  # context-only window
        self.assertFalse(stats['optimizer_step'])
        self.assertTrue(torch.equal(before, model.feature_memory.weight))


if __name__ == '__main__':
    unittest.main()

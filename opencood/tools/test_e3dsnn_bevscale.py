import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from opencood.models.sub_modules.e3dsnn.bev_scale import ScaledCount4, scale_report, add_unit_scales
from opencood.tools.launch_e3dsnn_bevscale_round import make_jobs, source_gate, GROUPS, contrasts


class Count(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.floor(x.clamp(0, 4) + .5)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * ((x > 0) & (x < 4)).to(g.dtype)


class Neuron(nn.Module):
    def forward(self, x): return Count.apply(x)


class ScaleTest(unittest.TestCase):
    def test_unit_scale_preserves_values_and_input_surrogate(self):
        for learnable in (False, True):
            module = ScaledCount4(Neuron(), learnable)
            x = torch.tensor([-1., 0., .49, .5, 1.3, 3.8, 4., 5.], requires_grad=True)
            y = x.detach().clone().requires_grad_()
            self.assertTrue(torch.equal(module(x), Neuron()(y)))
            module(x).sum().backward(); Neuron()(y).sum().backward()
            self.assertTrue(torch.equal(x.grad, y.grad))
            if learnable:
                expected = (Neuron()(y) - y * ((y > 0) & (y < 4))).sum()
                self.assertAlmostEqual(float(module.log_scale.grad), float(expected), places=5)
            else: self.assertIsNone(module.log_scale.grad)

    def test_bounded_step_and_five_levels(self):
        module = ScaledCount4(Neuron(), True)
        for log in (-100., -.8, 0., .9, 100.):
            with torch.no_grad(): module.log_scale.fill_(log)
            scale = module.scale(); self.assertGreaterEqual(float(scale), .25); self.assertLessEqual(float(scale), 4.)
            counts = module(torch.linspace(-3, 20, 200)) / scale
            torch.testing.assert_close(counts, counts.round(), rtol=0, atol=1e-6)
            self.assertGreaterEqual(float(counts.min()), 0); self.assertLessEqual(float(counts.max()), 4)

    def test_scalar_gain_can_fold_into_both_convolutions(self):
        for conv in (nn.Conv2d(3, 2, 3, bias=False), nn.ConvTranspose2d(3, 2, 2, stride=2, bias=False)):
            other = copy.deepcopy(conv); gain = 1.7
            with torch.no_grad(): other.weight.mul_(gain)
            counts = torch.randint(0, 5, (2, 3, 7, 9)).float()
            torch.testing.assert_close(conv(gain * counts), other(counts), rtol=1e-5, atol=1e-5)

    def test_state_extension_is_restricted_and_non_aliasing(self):
        source = {'reg_head.weight': torch.ones(2)}
        target = dict(source, **{'backbone_2d.blocks.{}.log_scale'.format(i): torch.zeros(()) for i in range(14)})
        merged = add_unit_scales(source, target)
        merged['reg_head.weight'].zero_(); self.assertTrue(torch.equal(source['reg_head.weight'], torch.ones(2)))
        target['other'] = torch.zeros(())
        with self.assertRaises(ValueError): add_unit_scales(source, target)

    def test_fixed_buffer_learned_parameter_and_scale_audit(self):
        fixed, learned = [ScaledCount4(Neuron(), x) for x in (False, True)]
        self.assertEqual(len(list(fixed.parameters())), 0)
        self.assertEqual(len(list(learned.parameters())), 1)
        optimizer = torch.optim.SGD(learned.parameters(), lr=.01)
        learned(torch.tensor([.7, 4.5])).sum().backward(); optimizer.step()
        self.assertEqual(scale_report(fixed)['changed_from_one'], 0)
        self.assertEqual(scale_report(learned)['changed_from_one'], 1)
        with torch.no_grad(): learned.log_scale.fill_(float('inf'))
        with self.assertRaises(ValueError): scale_report(learned)

    def test_equal_budget_shared_initializer_and_effect_units(self):
        for group in GROUPS:
            _, jobs = make_jobs('/experiment', group, 'adapt10')
            cmd = jobs[0]['command']; self.assertEqual(cmd[cmd.index('--epochs') + 1], '10')
            self.assertEqual(cmd[cmd.index('--checkpoint') + 1], str(Path('/experiment/initial.pth')))
            self.assertNotIn('--warmup-epochs', cmd)
            self.assertEqual(len(jobs), 6)
        values = contrasts(dict(fixed_high=.50, learned_high=.52, fixed_low=.51, learned_low=.54))
        self.assertAlmostEqual(values['learned_at_high_lr'], 2.)
        self.assertAlmostEqual(values['learned_at_low_lr'], 3.)

    def test_no_launch_before_receiver_round_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); (p / 'round_status.json').write_text(json.dumps(dict(state='running')))
            with self.assertRaises(AssertionError): source_gate(p)


if __name__ == '__main__': unittest.main()

import unittest
import torch
from opencood.models.sub_modules.ectra_recurrent_alignment import EctraRecurrentAlignment
from opencood.tools.ectra_gradient_diagnostics import EctraInputGradientProbe


class InputGradientTest(unittest.TestCase):
    def test_probe_preserves_forward_and_parameter_gradients(self):
        torch.manual_seed(7)
        module = EctraRecurrentAlignment({'feature_dim': 2, 'hidden_dim': 4}).eval()
        x = torch.rand(4, 2, 4, 5)
        lengths = torch.tensor([2])
        times = torch.tensor([0., 0., -3., -6.])
        baseline, _ = module(x, lengths, times)
        baseline.sum().backward()
        expected = {k: p.grad.clone() for k, p in module.named_parameters() if p.grad is not None}
        module.zero_grad()
        probe = EctraInputGradientProbe(module)
        actual, _ = module(x, lengths, times)
        torch.testing.assert_allclose(actual, baseline)
        rows = probe.summarize(actual.sum())
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row['connected'])
            self.assertIn('ego', row['branches'])
            for metrics in row['branches'].values():
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        actual.sum().backward()
        for name, parameter in module.named_parameters():
            if name in expected:
                torch.testing.assert_allclose(parameter.grad, expected[name])
        probe.close()
        self.assertFalse(probe.records)
        self.assertFalse(probe.handles)


if __name__ == '__main__':
    unittest.main()

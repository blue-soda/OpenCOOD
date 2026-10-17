import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_readout import residual_readout


class ReadoutTest(unittest.TestCase):
    def test_continuous_preserves_small_signed_updates_and_gradients(self):
        ego = torch.tensor([0., 1., 4., 2.]).reshape(1, 1, 1, 4)
        residual = torch.tensor([-.2, .2, .7, .3]).reshape_as(ego).requires_grad_()
        support = torch.tensor([1., 1., 1., 0.]).reshape_as(ego)
        quantize = lambda x: x.clamp(0, 4).add(.5).floor()
        discrete = residual_readout(ego, residual, support, quantize)
        continuous = residual_readout(ego, residual, support, quantize, 'continuous')
        self.assertTrue(torch.equal(discrete, ego))
        torch.testing.assert_close(continuous, torch.tensor([-.2, 1.2, 4.7, 2.]).reshape_as(ego))
        continuous.sum().backward()
        self.assertTrue(torch.equal(residual.grad, support))

    def test_zero_residual_identity_and_legacy_relu(self):
        ego = torch.tensor([0., 1., 2., 4.]).reshape(1, 1, 1, 4)
        mask = torch.ones_like(ego)
        out = residual_readout(ego, torch.zeros_like(ego), mask, torch.relu, 'continuous')
        self.assertTrue(torch.equal(out, ego))
        residual = torch.full_like(ego, -.5)
        self.assertTrue(torch.equal(residual_readout(ego, residual, mask, torch.relu), torch.relu(ego + residual)))


if __name__ == '__main__':
    unittest.main()

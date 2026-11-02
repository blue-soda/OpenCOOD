import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_spatial_basis_readout import spatial_basis_conv, spatial_basis_readout


class SpatialBasisTests(unittest.TestCase):
    def test_full_basis_preserves_channel_order_stride_dilation_bias(self):
        torch.manual_seed(702)
        for outputs in (5, 12):
            for stride, dilation in ((1, 1), (2, 2)):
                conv = nn.Conv2d(3, outputs, 3, stride, padding=dilation, dilation=dilation).double()
                factored = spatial_basis_conv(conv, min(outputs, 9))
                x = torch.randn(2, 3, 17, 19, dtype=torch.float64)
                self.assertTrue(torch.allclose(conv(x), factored(x), atol=1e-12, rtol=1e-11))

    def test_reduced_basis_preserves_parent_and_nonlinearity_and_gradients(self):
        source = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()).eval()
        before = {k: v.clone() for k, v in source.state_dict().items()}
        result, changes = spatial_basis_readout(source, 4)
        self.assertEqual(changes[0]['rank'], 4)
        self.assertLess(sum(p.numel() for p in result.parameters()), sum(p.numel() for p in source.parameters()))
        self.assertTrue(torch.equal(result[1].running_var, source[1].running_var))
        result(torch.randn(2, 8, 11, 13)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in result.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in source.state_dict().items()))

    def test_rank_validation(self):
        for rank in (0, 10, True, 2.5):
            with self.assertRaises(ValueError):
                spatial_basis_conv(nn.Conv2d(3, 16, 3), rank)


if __name__ == '__main__':
    unittest.main()

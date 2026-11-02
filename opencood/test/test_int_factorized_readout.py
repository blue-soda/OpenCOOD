import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_factorized_readout import factorize_conv,compact_readout


class FactorizedReadoutTests(unittest.TestCase):
    def test_full_rank_preserves_spatial_bias_stride_and_padding(self):
        torch.manual_seed(321)
        for stride in (1,2):
            conv=nn.Conv2d(8,12,3,stride,padding=1);model=factorize_conv(conv,12);x=torch.randn(2,8,13,17)
            self.assertTrue(torch.allclose(conv(x),model(x),atol=3e-6,rtol=1e-4))

    def test_rank_reduction_preserves_nonlinear_layers_and_trainable_gradients(self):
        source=nn.Sequential(nn.Conv2d(8,12,3,padding=1),nn.BatchNorm2d(12),nn.ReLU());source.eval();before={k:v.clone() for k,v in source.state_dict().items()}
        model,changed=compact_readout(source,4);self.assertEqual(changed[0]['rank'],4)
        self.assertLess(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in source.parameters()))
        model(torch.randn(2,8,13,17)).square().mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(torch.equal(v,source.state_dict()[k]) for k,v in before.items()))


if __name__=='__main__':unittest.main()

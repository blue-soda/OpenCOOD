import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_compact_readout import factorize_input,compact_shrink
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class CompactReadoutTests(unittest.TestCase):
    def test_checkpoint_layout_keeps_bn_and_full_rank_output(self):
        source=DownsampleConv(dict(input_dim=12,kernal_size=[3],dim=[17],stride=[1],padding=[1])).eval()
        model,energy=compact_shrink(source,17);model.eval()
        x=torch.randn(1,12,9,13)
        self.assertEqual(len(energy),2)
        self.assertIsInstance(model.layers[0].double_conv[1],nn.BatchNorm2d)
        self.assertIsInstance(model.layers[0].double_conv[4],nn.BatchNorm2d)
        self.assertTrue(torch.allclose(source(x),model(x),atol=1e-5,rtol=1e-5))

    def test_full_rank_preserves_bias_and_boundary_values(self):
        torch.manual_seed(19)
        source=nn.Conv2d(12,17,3,padding=1)
        model,energy=factorize_input(source,12)
        x=torch.randn(1,12,9,13)
        self.assertAlmostEqual(energy,1.)
        self.assertTrue(torch.allclose(source(x),model(x),atol=1e-5,rtol=1e-5))

    def test_low_rank_local_crops_and_gradient(self):
        model,energy=factorize_input(nn.Conv2d(12,17,3,padding=1),5)
        model.eval()
        self.assertGreater(energy,0.);self.assertLess(energy,1.)
        x=torch.randn(1,12,17,21,requires_grad=True)
        old=torch.zeros(1,17,17,21);mask=torch.zeros(1,1,17,21,dtype=torch.bool)
        mask[...,0,0]=True;mask[...,-1,-1]=True
        actual,written,_=refresh_tiles(model,x,old,mask,1,1,4,allow_fallback=False)
        reference=model(x)
        self.assertTrue(torch.allclose(actual[written.expand_as(actual)],reference[written.expand_as(reference)],atol=1e-5,rtol=1e-5))
        actual.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()

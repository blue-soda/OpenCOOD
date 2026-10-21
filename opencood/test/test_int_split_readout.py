import unittest
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout


class SplitReadoutTests(unittest.TestCase):
    def make_source(self):
        torch.manual_seed(41)
        source=DownsampleConv(dict(input_dim=12,kernal_size=[3],dim=[8],stride=[1],padding=[1])).eval()
        for layer in source.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):
                layer.running_mean.normal_();layer.running_var.uniform_(.5,2.)
                layer.bias.data.normal_()
        return compact_shrink(source,4)[0].eval()

    def test_complete_split_preserves_boundary_outputs_and_parameter_count(self):
        source=self.make_source();split=SplitCompactReadout(source,4).eval()
        x=torch.randn(1,12,9,13)
        self.assertTrue(torch.allclose(source(x),split(x),atol=1e-6,rtol=1e-5))
        self.assertEqual(sum(p.numel() for p in source.parameters()),sum(p.numel() for p in split.parameters()))
        x.requires_grad_(True)
        split(x).square().mean().backward()
        self.assertGreater(float(x.grad[:,:4].abs().sum()),0.)

    def test_bias_free_deep_projection_commutes_with_bilinear_warp(self):
        split=SplitCompactReadout(self.make_source(),4).eval()
        old=torch.randn(1,8,9,13)
        theta=torch.tensor([[[.97,-.1,.15],[.1,.97,-.08]]])
        grid=F.affine_grid(theta,old.shape,align_corners=False)
        warp=lambda x:F.grid_sample(x,grid,align_corners=False,padding_mode='zeros')
        self.assertTrue(torch.allclose(warp(split.history(old)),split.history(warp(old)),atol=1e-6,rtol=1e-5))


if __name__=='__main__':unittest.main()

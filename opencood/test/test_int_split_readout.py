import unittest
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout,diagnostic_repair_selection
from opencood.models.sub_modules.int_light_repair import LightRepair


class SplitReadoutTests(unittest.TestCase):
    def test_diagnostic_selection_keeps_budget_and_uncovered_boundary(self):
        old=torch.zeros(1,2,7,9);candidate=old.clone()
        candidate[:,:,4:7,4:8]=3.;candidate[:,:,:4,:4]=1.
        valid=torch.ones_like(old[:,:1],dtype=torch.bool);valid[:,:,:,-1]=False
        request=~valid;request[:,:,:4,:4]=True
        selected,stats=diagnostic_repair_selection(candidate,old,request,valid,'energy')
        self.assertEqual(stats['selected_tiles'],stats['change_tiles'])
        self.assertEqual(stats['uncovered_tiles'],2)
        self.assertTrue(torch.equal(selected[:,:,4:7,4:8],candidate[:,:,4:7,4:8]))
        self.assertEqual(float(selected[:,:,:4,:4].sum()),0.)
        changed,_=diagnostic_repair_selection(candidate,old,request,valid,'change')
        other,_=diagnostic_repair_selection(candidate,old,request,valid,'complement')
        self.assertTrue(torch.equal(changed+other,candidate))  # Uncovered cells have zero residual here.
        self.assertEqual(float(old.sum()),0.)

    def test_diagnostic_empty_and_complete_masks(self):
        old=torch.zeros(1,2,5,7);candidate=torch.ones_like(old)
        valid=torch.ones_like(old[:,:1],dtype=torch.bool)
        for filled in (False,True):
            mask=torch.full_like(valid,filled)
            out,stats=diagnostic_repair_selection(candidate,old,mask,valid,'energy')
            self.assertTrue(torch.equal(out,candidate if filled else old))
            self.assertEqual(stats['executed_input_area_ratio'],1.)

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

    def test_zero_correction_preserves_cache_and_learns_through_frozen_readout(self):
        split=SplitCompactReadout(self.make_source(),4).eval()
        for p in split.parameters():p.requires_grad_(False)
        repair=LightRepair(4,width=8).eval()
        evidence=torch.randn(1,128,9,13)
        old=torch.randn(1,4,9,13);saved=old.clone()
        current=torch.randn_like(old)
        packed=torch.cat([evidence,old,torch.ones_like(old[:,:1])],1)
        corrected=repair(packed)
        self.assertTrue(torch.equal(corrected,old))
        self.assertTrue(torch.equal(split.fuse(current,corrected),split.fuse(current,old)))
        split.fuse(current,corrected).square().mean().backward()
        self.assertGreater(float(repair.net[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in split.parameters()))
        self.assertTrue(torch.equal(old,saved))


if __name__=='__main__':unittest.main()

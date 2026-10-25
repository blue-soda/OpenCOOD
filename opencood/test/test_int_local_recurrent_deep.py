import unittest
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.models.sub_modules.int_local_recurrent_deep import local_repair,LocalRecurrentDeepState


class TestLocalRecurrentDeep(unittest.TestCase):
    @torch.no_grad()
    def test_selected_matches_dense_and_preserves_unselected(self):
        torch.manual_seed(31);repair=LightRepair(96,8).eval()
        torch.nn.init.normal_(repair.net[-1].weight,std=.1)
        h=torch.randn(1,128,9,11);old=torch.randn(1,96,9,11);original=old.clone();valid=torch.ones(1,1,9,11,dtype=torch.bool)
        for kind in ('empty','full','corners','middle'):
            mask=torch.zeros_like(valid)
            if kind=='full':mask.fill_(True)
            if kind=='corners':mask[...,0,0]=True;mask[...,-1,-1]=True
            if kind=='middle':mask[...,3:6,4:7]=True
            actual,info=local_repair(repair,h,old,valid,mask)
            dense,reference=local_repair(repair,h,old,valid,mask,'masked_dense')
            self.assertTrue(torch.allclose(actual,dense,atol=2e-6,rtol=2e-6))
            self.assertTrue(torch.equal(actual.masked_select(~mask),old.masked_select(~mask)))
            self.assertTrue(torch.equal(old,original))
            support=F.max_pool2d(mask.float(),3,1,1)>0
            expected=int(support.sum())*repair.net[0].weight.numel()+int(mask.sum())*(repair.net[2].weight.numel()+repair.net[4].weight.numel())
            full=mask.numel()*sum(repair.net[i].weight.numel() for i in (0,2,4))
            self.assertEqual(info['deep_macs_saved'],full-expected)
            if kind!='full':self.assertEqual(info['deep_extra_linear_macs'],expected)
            else:self.assertTrue(torch.equal(actual,dense))

    @torch.no_grad()
    def test_uncovered_cells_are_repaired_even_without_h_change(self):
        repair=LightRepair(96,8).eval();repair.net[-1].bias.fill_(1.)
        cache=LocalRecurrentDeepState(repair,[-2,-2,-1,2,2,1])
        h=torch.zeros(1,128,5,5);old=torch.zeros(1,96,5,5);valid=torch.ones(1,1,5,5,dtype=torch.bool);valid[...,0,0]=False
        value,info=cache.update(h,old,valid,torch.zeros_like(valid))
        self.assertEqual(float(value[...,0,0].mean()),1.)
        self.assertEqual(float(value[...,1:,1:].sum()),0.)
        self.assertEqual(info['deep_write_fraction'],1/25.)

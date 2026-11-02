import copy
import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_causal_readout_qat import CausalReadoutQAT
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout


class CausalQATTests(unittest.TestCase):
    def test_ann_flattening_and_frozen_bn(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval();model=CausalReadoutQAT(source,[[.125]*2,[.125]*3],False);x=torch.randn(1,2,4,5)
        self.assertTrue(torch.equal(source(x),model(x)));self.assertEqual(model.last_reference_flops,0)
        model.train()
        with self.assertRaises(ValueError):model(x)

    def test_surrogate_gradients_and_exact_reexport_after_update(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval();q=[[.125]*2,[.125]*3];model=CausalReadoutQAT(source,q,True);x=torch.randn(1,2,4,5);bn_before=model.layers[1].running_mean.clone()
        for step in range(2):
            y=model(x);reference=IntegerReadout(nn.Sequential(*copy.deepcopy(list(model.layers))),q,False)
            with torch.no_grad():self.assertTrue(torch.equal(y.detach(),reference(x)))
            y.square().mean().backward();self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()));self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0)
            self.assertTrue(torch.equal(bn_before,model.layers[1].running_mean));self.assertTrue(all(v<2**53 for v in model.last_bounds))
            with torch.no_grad():
                for p in model.parameters():p.add_(p.grad,alpha=-1e-4)
            model.zero_grad(set_to_none=True)


if __name__=='__main__':unittest.main()

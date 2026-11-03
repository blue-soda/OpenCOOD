import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_causal_readout_qat import CausalReadoutQAT
from opencood.models.sub_modules.int_joint_state_qat import FrozenFineObservation
from opencood.models.sub_modules.resblock import BasicBlock


class JointStateQATTests(unittest.TestCase):
    def test_frozen_readout_passes_surrogate_gradient_to_state(self):
        torch.manual_seed(303)
        source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval()
        # Existing readout wrapper accepts the sequential layer collection.
        model=CausalReadoutQAT(source,[[.125,.25],[.125]*3],True).eval()
        for p in model.parameters():p.requires_grad_(False)
        x=torch.randn(1,2,5,7,requires_grad=True);y=model(x);y.square().mean().backward()
        self.assertIsNotNone(x.grad);self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.no_grad():self.assertTrue(torch.equal(y.detach(),model(x.detach())))

    def test_fine_freeze_boundary_and_later_state_gradient(self):
        torch.manual_seed(303)
        source=nn.Sequential(BasicBlock(2,4,2,nn.Sequential(nn.Conv2d(2,4,1,stride=2,bias=False),nn.BatchNorm2d(4))),BasicBlock(4,4),BasicBlock(4,4)).eval()
        calibration={n:dict(quanta=[.125]*m.in_channels) for n,m in source.named_modules() if isinstance(m,nn.Conv2d)}
        fine=FrozenFineObservation(source,calibration,'dense').eval()
        for p in fine.parameters():p.requires_grad_(False)
        x=torch.randn(1,2,8,10,requires_grad=True);features=fine(x);self.assertFalse(features.requires_grad)
        update=nn.Conv2d(4,3,1);update(features).square().mean().backward()
        self.assertIsNone(x.grad);self.assertTrue(all(p.grad is None for p in fine.parameters()));self.assertGreater(float(update.weight.grad.abs().sum()),0)
        next(fine.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):fine(x)


if __name__=='__main__':unittest.main()

"""Functional tests for the larger receiver; no AP claim from synthetic input."""
import copy
import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_pyramid_current import PyramidCurrent
from opencood.models.fuse_modules.e3dsnn_progressive import count_readout
from opencood.utils.cp_optimizer import apply_warmup


class PyramidTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.u = torch.rand(1,128,2,13,17)*3
        self.d = torch.randn_like(self.u)
        self.mask = torch.zeros(1,1,2,13,17, dtype=torch.bool)
        self.mask[...,6,8] = True
        self.coverage = torch.ones_like(self.mask, dtype=torch.float32)

    def test_identity_and_deterministic_initialization(self):
        rng = torch.get_rng_state().clone()
        a, b = PyramidCurrent(2), PyramidCurrent(2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        base = self.d*.1
        current, support = a(self.u,self.d,base,self.mask,self.coverage)
        actual = torch.where(support,count_readout(self.u+current),count_readout(self.u))
        expected = torch.where(self.mask,count_readout(self.u+base),count_readout(self.u))
        self.assertTrue(torch.equal(actual,expected))
        self.assertGreater(int(support.sum()),int(self.mask.sum()))

    def test_empty_and_out_of_range_remain_local_after_learning(self):
        a=PyramidCurrent(2)
        with torch.no_grad():a.output.weight.fill_(.01)
        zero = torch.zeros_like(self.mask)
        for mask,coverage in [(zero,self.coverage),(self.mask,self.coverage*0)]:
            # Caller ensures native received mask excludes geometric noncoverage.
            mask=mask & (coverage>0)
            current,support=a(self.u,self.d,self.d*.1,mask,coverage)
            self.assertEqual(float(current.abs().sum()),0)
            self.assertFalse(bool(support.any()))

    def test_two_step_gradients_and_spatial_effect(self):
        a=PyramidCurrent(2);opt=torch.optim.Adam(a.parameters(),lr=1e-3)
        for step in range(2):
            opt.zero_grad()
            current,support=a(self.u,self.d,self.d*.1,self.mask,self.coverage)
            (current-.2).square().mean().backward()
            self.assertGreater(float(a.output.weight.grad.abs().sum()),0)
            for name,p in a.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                if step==1:self.assertGreater(float(p.grad.abs().sum()),0,name)
            opt.step()
        current,_=a(self.u,self.d,self.d*.1,self.mask,self.coverage)
        self.assertGreater(float((current*(~self.mask)).abs().sum()),0)

    def test_warmup_and_resume_multistep(self):
        p=torch.nn.Parameter(torch.ones(1));o=torch.optim.Adam([p],lr=.001)
        s=torch.optim.lr_scheduler.MultiStepLR(o,[2],gamma=.1)
        for i in range(4):
            apply_warmup(o,0,i,4,1);self.assertAlmostEqual(o.param_groups[0]['lr'],.001*(i+1)/4)
            o.step()
        s.step();state=copy.deepcopy(o.state_dict());sched=copy.deepcopy(s.state_dict())
        o2=torch.optim.Adam([p],lr=.001);s2=torch.optim.lr_scheduler.MultiStepLR(o2,[2],gamma=.1)
        o2.load_state_dict(state);s2.load_state_dict(sched)
        apply_warmup(o2,1,0,4,1);o2.step();s2.step()
        self.assertAlmostEqual(o2.param_groups[0]['lr'],.0001)


if __name__=='__main__':unittest.main()

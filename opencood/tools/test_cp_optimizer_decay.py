"""Reproduce coupled-Adam decay of task-inactive cold-start hidden weights."""
import copy
import unittest
import torch
from opencood.utils.cp_optimizer import make_optimizer,new_parameter_health


class ColdModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Parameter(torch.ones(1)*.01)
        self.branch=torch.nn.Linear(4,4,bias=False)
        torch.nn.init.constant_(self.branch.weight,.01)


class DecayTest(unittest.TestCase):
    def test_cold_start_preservation_and_unchanged_base(self):
        cfg={'optimizer':dict(core_method='Adam',lr=5e-5,new_parameter_prefix='branch.',
            new_parameter_lr=1e-3,args=dict(weight_decay=1e-4,eps=1e-8))}
        old=ColdModule();new=copy.deepcopy(old)
        first=make_optimizer(cfg,old)
        revised=copy.deepcopy(cfg);revised['optimizer']['new_parameter_weight_decay']=0.
        second=make_optimizer(revised,new)
        for _ in range(1500):
            for model,opt in [(old,first),(new,second)]:
                opt.zero_grad();(model.base.sum()+model.branch.weight.sum()*0).backward();opt.step()
        self.assertTrue(torch.equal(old.base,new.base))
        self.assertLess(float(old.branch.weight.abs().max()),1e-12)
        self.assertTrue(torch.equal(new.branch.weight,torch.full_like(new.branch.weight,.01)))
        self.assertEqual(first.param_groups[1]['weight_decay'],1e-4)
        self.assertEqual(second.param_groups[0]['weight_decay'],1e-4)
        self.assertEqual(new_parameter_health(second)['tensors_below_1e_8'],0)

    def test_reject_invalid_decay(self):
        for value in (-1,float('nan'),float('inf')):
            cfg={'optimizer':dict(core_method='Adam',lr=5e-5,new_parameter_prefix='branch.',
                new_parameter_lr=1e-3,new_parameter_weight_decay=value)}
            with self.assertRaises(ValueError):make_optimizer(cfg,ColdModule())


if __name__=='__main__':unittest.main()

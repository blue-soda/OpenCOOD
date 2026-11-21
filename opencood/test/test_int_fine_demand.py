import unittest

import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_fine_demand import fine_support, QuotaFineEdge


class FineDemandTest(unittest.TestCase):
    def test_support_matches_operator_derivative(self):
        for h,w in [(17,20),(18,19),(17,19),(18,20)]:
            fine=torch.zeros(1,1,h,w,requires_grad=True)
            kernel=torch.ones(1,1,3,3)
            x=F.conv2d(fine,kernel,stride=2,padding=1)
            x=F.conv2d(F.conv2d(x,kernel,padding=1),kernel,padding=1)
            for positions in ([],[(0,0)],[(-1,-1)],[(2,3),(5,7)]):
                active=torch.zeros_like(x,dtype=torch.bool)
                for y,z in positions:
                    active[...,y,z]=True
                derivative=torch.autograd.grad((x*active).sum(),fine,retain_graph=True)[0]
                self.assertTrue(torch.equal(derivative!=0,fine_support(active,(h,w))))

    def test_dropping_fraction_keeps_untransmitted_whole_levels(self):
        # This is the isolated state writeback after ordinary receive.
        value=torch.tensor([[[[1.10,-1.10,0.13]]]])
        for mode in ('unit_carry','unit_round'):
            sender=QuotaFineEdge([0,0,0,1,1,1],mode,1)
            sender.phase='ordinary'
            sender.levels=torch.tensor([[[[1,0,0]]]],dtype=torch.int)
            sender.read=torch.tensor([[[[True,True,False]]]])
            state={'value':value.clone()}
            sender.commit(state,False)
            expected=value.clone()
            if mode=='unit_round':
                expected[...,0]=1.
                expected[...,1]=-1.
            self.assertTrue(torch.equal(state['value'],expected))
            self.assertEqual(int(state['fine_levels'].sum()),1)


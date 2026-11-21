import unittest

import torch
from torch import nn

from opencood.models.sub_modules.int_branch_fusion import BranchFusion


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class ThreeBranches(unittest.TestCase):
    @torch.no_grad()
    def test_disjoint_dirty_neighborhoods_and_refresh(self):
        torch.manual_seed(303)
        m = BranchFusion(nn.Conv2d(384,128,3,padding=1).cuda().eval(), verify=True)
        x = torch.zeros(1,384,9,10,device='cuda')
        x[:,0,0,0] = 1
        x[:,128,-1,-1] = 2
        x[:,256,4,4] = 3
        m.begin(None,'initial')
        m(x)
        self.assertEqual([b['executed_taps'] for b in m.ledger['branches']],[4,4,9])
        state = dict(branch_fusion=m.new)
        x[:,0,0,0] = 0
        x[:,128,-1,-1] = 4
        m.begin(state,None)
        y = m(x)
        self.assertEqual(m.ledger['joint_dirty'],8)
        self.assertEqual([b['dirty'] for b in m.ledger['branches']],[4,4,0])
        m.begin(dict(branch_fusion=m.new),None)
        self.assertTrue(torch.equal(m(x),y))
        self.assertEqual(m.ledger['float_conv'],0)
        m.begin(dict(branch_fusion=m.new),'period')
        self.assertTrue(torch.equal(m(x),y))
        self.assertEqual(m.ledger['joint_dirty'],90)

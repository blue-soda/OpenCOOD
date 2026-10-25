import unittest
import torch
from opencood.tools.audit_int_stem_reuse import dependency_cost,fit_stem_threshold,score_mask


class TestStemAudit(unittest.TestCase):
    def test_dependency_cost_and_protected_fit(self):
        weights=dict(stem=[2,3,1],u=[(1,4)]*4)
        mask=torch.ones(1,1,20,24,dtype=torch.bool)
        self.assertEqual(dependency_cost(mask,weights),(20*24*20,20*24*6))
        self.assertEqual(dependency_cost(~mask,weights),(0,0))
        center=torch.zeros_like(mask);center[:,:,10,12]=True
        # Radii0..5 have areas1,9,25,49,81,121 away from borders.
        self.assertEqual(dependency_cost(center,weights),
            (sum(p+4*m for p,m in zip([81,49,25,9],[49,25,9,1])),121*2+81*4))
        torch.manual_seed(303);scores=torch.rand(2,1,5,6)
        mandatory=torch.zeros(2,1,20,24,dtype=torch.bool);mandatory[:,:,:2,:2]=True
        target=dependency_cost(score_mask(scores,mandatory,.6),weights)[0]
        threshold,actual=fit_stem_threshold(scores,mandatory,weights,target)
        self.assertLessEqual(actual,target)
        self.assertEqual(actual,dependency_cost(score_mask(scores,mandatory,threshold),weights)[0])
        self.assertTrue(bool((score_mask(scores,mandatory,threshold)|~mandatory).all()))

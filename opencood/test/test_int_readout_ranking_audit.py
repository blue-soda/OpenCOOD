import unittest
import torch
from opencood.tools.audit_int_readout_ranking import executed_macs,budgeted_ranking


class TestBudgetedRanking(unittest.TestCase):
    def test_nested_prefix_budget_and_mandatory(self):
        scores=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        mandatory=torch.zeros(1,1,16,16,dtype=torch.bool);mandatory[:,:,:2,:2]=True
        minimum=executed_macs(mandatory,10,9)
        maximum=16*16*19
        previous=mandatory
        for budget in (minimum,minimum+250,maximum//2,maximum):
            mask,stats=budgeted_ranking(scores,mandatory,budget,10,9)
            self.assertLessEqual(stats['macs'],budget)
            self.assertEqual(stats['macs'],executed_macs(mask,10,9))
            self.assertTrue(bool((mask|~mandatory).all()))
            self.assertTrue(bool((mask|~previous).all()))
            previous=mask
        self.assertTrue(bool(mask.all()))
        self.assertEqual(stats['tiles'],16)
        with self.assertRaises(ValueError):budgeted_ranking(scores,mandatory,minimum-1,10,9)


if __name__=='__main__':unittest.main()

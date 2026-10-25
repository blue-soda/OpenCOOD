import unittest
import torch
from opencood.models.sub_modules.int_readout_calibration import fit_threshold,threshold_mask
from opencood.models.sub_modules.int_budgeted_readout import executed_macs


class TestReadoutCalibration(unittest.TestCase):
    def test_aggregate_cost_and_protection(self):
        torch.manual_seed(303)
        scores=torch.rand(5,1,8,9)
        protected=torch.zeros(5,1,32,34,dtype=torch.bool);protected[:,:,:2,:3]=True
        result=fit_threshold(scores,protected,5,3)
        self.assertEqual(result['status'],'FITTED')
        mask=threshold_mask(scores,protected,result['threshold'])
        self.assertEqual(executed_macs(mask,5,3),result['calibrated_macs'])
        self.assertLessEqual(result['calibrated_macs'],result['target_macs'])
        self.assertTrue(bool((mask|~protected).all()))
        self.assertTrue(bool((~mask|threshold_mask(scores,protected,.5)).all()))

    def test_unreachable_budget_is_not_silently_claimed(self):
        result=fit_threshold(torch.ones(2,1,3,3),torch.ones(2,1,10,12,dtype=torch.bool),5,3)
        self.assertEqual(result['status'],'UNREACHABLE')
        self.assertGreater(result['mandatory_macs'],result['target_macs'])

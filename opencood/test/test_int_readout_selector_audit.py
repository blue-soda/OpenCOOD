import unittest
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_local_cache import local_change_mask
from opencood.tools.audit_int_readout_selector import energy_masks


class TestSelectorAudit(unittest.TestCase):
    def test_baseline_identity_floor_subset_and_protection(self):
        current=torch.full((1,16,32,32),10.)
        current[:,:,:16]=.01
        reference=current.clone();reference[:,:,:16]=0
        valid=torch.ones(1,1,32,32,dtype=torch.bool);valid[:,:,-1,-1]=False
        confidence=torch.zeros(1,1,32,32);confidence[:,:,4,4]=.5
        original=current.clone();old=reference.clone()
        masks,groups,energy,error,quantiles,guards=energy_masks(current,reference,valid,confidence,.5)
        expected=(F.max_pool2d(local_change_mask(current,reference,valid,.5).float(),5,1,2)>0)|guards|~valid
        self.assertTrue(torch.equal(masks['baseline'][0],expected))
        self.assertLess(int(masks['floor_q50'][0].sum()),int(expected.sum()))
        for mask,_ in masks.values():
            self.assertTrue(bool((expected|~mask).all()))
            self.assertTrue(bool((mask|valid).all()))
            self.assertTrue(bool((mask|~guards).all()))
        self.assertEqual(torch.bincount(groups).tolist(),[16]*4)
        self.assertTrue(torch.isfinite(energy).all());self.assertTrue(torch.isfinite(error).all())
        self.assertTrue(torch.equal(current,original));self.assertTrue(torch.equal(reference,old))


if __name__=='__main__':unittest.main()

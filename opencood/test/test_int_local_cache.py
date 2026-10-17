import unittest
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_local_cache import affected_outputs,local_requests,local_change_mask


class LocalCacheTests(unittest.TestCase):
    def test_affected_outputs_matches_two_convolution_receptive_field(self):
        x=torch.zeros(1,1,30,34);x[...,10,12]=1;x[...,29,33]=1
        kernel=torch.ones(1,1,3,3)
        actual=F.conv2d(F.conv2d(x,kernel,stride=2,padding=1),kernel,padding=1)>0
        self.assertTrue(torch.equal(actual,affected_outputs(x.bool(),2,3)))

    def test_boundary_repair_propagates_to_next_stage(self):
        changed=torch.zeros(1,1,100,252,dtype=torch.bool)
        valid2=torch.ones(1,1,50,126,dtype=torch.bool)
        valid3=torch.ones(1,1,25,63,dtype=torch.bool)
        a,b=local_requests(changed,valid2,valid3,4)
        self.assertFalse(a.any());self.assertFalse(b.any())
        valid2[...,-1,-1]=False
        a,b=local_requests(changed,valid2,valid3,4)
        self.assertTrue(a[...,-1,-1]);self.assertTrue(b[...,-1,-1])
        self.assertGreater(int(b.sum()),int(a.sum()))
        self.assertTrue(bool((b|affected_outputs(a,2,19)).eq(b).all()))

    def test_local_change_preserves_ragged_edges_and_new_view(self):
        x=torch.ones(1,16,25,63);valid=torch.ones(1,1,25,63,dtype=torch.bool)
        self.assertFalse(local_change_mask(x,x,valid,.1).any())
        valid[...,-1,-1]=False
        mask=local_change_mask(x,x,valid,.1)
        self.assertEqual(tuple(mask.shape),(1,1,25,63));self.assertEqual(int(mask.sum()),1)
        self.assertTrue(local_change_mask(x,2*x,valid,.4).all())


if __name__=='__main__':unittest.main()

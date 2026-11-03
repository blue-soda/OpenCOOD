import unittest
import torch
from opencood.models.sub_modules.int_threshold_reference import IntegerThresholdMap


class ThresholdReferenceTests(unittest.TestCase):
    def test_signed_thresholds_ties_constants_and_fallback(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with torch.no_grad():
                sign=torch.tensor([1,-1,1],device=device,dtype=torch.int8)
                def oracle(acc):
                    factor=torch.tensor([.5,-.25,0.],device=device).reshape(1,3,1,1)
                    bias=torch.tensor([0.,.5,2.],device=device).reshape(1,3,1,1)
                    return torch.round(acc.float()*factor+bias).to(torch.int64)
                decoder=IntegerThresholdMap(oracle,sign,(8,8),levels=8,bound=63)
                values=torch.arange(-32,32,device=device).reshape(1,1,8,8).expand(1,3,8,8)
                got,stats=decoder.decode(values);self.assertTrue(torch.equal(got,oracle(values)));self.assertGreater(stats['fallback_elements'],0)
                values=values.clamp(-8,8);got,stats=decoder.decode(values);self.assertTrue(torch.equal(got,oracle(values)));self.assertEqual(stats['fallback_elements'],0)
                self.assertEqual(decoder.table_bytes,3*18*4+3)

    def test_relu_bound_and_grid_guard(self):
        with torch.no_grad():
            sign=torch.ones(1,dtype=torch.int8)
            oracle=lambda x:torch.round(x.float().clamp_min(0)/2).to(torch.int64)
            decoder=IntegerThresholdMap(oracle,sign,(8,8),levels=16,bound=31)
            values=torch.arange(-32,32).reshape(1,1,8,8);got,stats=decoder.decode(values)
            self.assertTrue(torch.equal(got,oracle(values)));self.assertEqual(stats['fallback_elements'],1)
            with self.assertRaises(ValueError):IntegerThresholdMap(oracle,sign,(2,2),levels=16,bound=31)

    def test_batch_norm_negative_gamma_exact_integer_domain(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            with torch.no_grad():
                bn=torch.nn.BatchNorm2d(3).to(device).eval()
                for key,values in [('weight',[-.7,.4,0.]),('bias',[.1,-.2,.3]),('running_mean',[.123,-.32,.2]),('running_var',[.7,1.2,.3])]:
                    getattr(bn,key).copy_(torch.tensor(values,device=device))
                bias=torch.tensor([.02,-.1,.03],device=device).reshape(1,3,1,1)
                def oracle(acc):return torch.round(torch.relu(bn((acc.double()*.03).float()+bias)).double()/.125).to(torch.int64)
                decoder=IntegerThresholdMap(oracle,torch.tensor([-1,1,1],device=device),(8,16),levels=32,bound=63)
                values=torch.cat((torch.arange(-63,64,device=device),torch.zeros(1,device=device,dtype=torch.int64))).reshape(1,1,8,16).expand(1,3,8,16)
                actual,stats=decoder.decode(values);self.assertTrue(torch.equal(actual,oracle(values)));self.assertEqual(stats['fallback_elements'],0)

if __name__=='__main__':unittest.main()

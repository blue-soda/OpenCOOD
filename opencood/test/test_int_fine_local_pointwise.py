import unittest
import torch
from opencood.models.sub_modules.int_fine_local_pointwise import local_bn_relu


class FineLocalPointwiseTests(unittest.TestCase):
    def test_masked_values_and_no_previous_mutation(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(303);bn=torch.nn.BatchNorm2d(4).to(device).eval()
            with torch.no_grad():
                bn.weight.copy_(torch.tensor([.4,-.7,0.,1.2],device=device));bn.bias.copy_(torch.tensor([.1,-.2,.3,.2],device=device));bn.running_var.copy_(torch.tensor([.7,1.3,.5,.8],device=device))
                x=torch.randn(1,4,7,11,device=device);old,stats=local_bn_relu(x,bn);saved=old.clone()
                mask=torch.zeros(1,1,7,11,dtype=torch.bool,device=device);mask[:,:,1:4,2:5]=True
                new=x+mask*torch.randn_like(x);got,stats=local_bn_relu(new,bn,old,mask)
                self.assertTrue(torch.equal(got,torch.relu(bn(new))));self.assertTrue(torch.equal(old,saved))
                self.assertEqual(stats['processed_bn_elements'],36);self.assertEqual(stats['output_cache_bytes'],4*4*7*11)

    def test_empty_support_skips_bn_and_full_reset(self):
        bn=torch.nn.BatchNorm2d(3).eval();calls=[];handle=bn.register_forward_hook(lambda m,i,o:calls.append(i[0].numel()))
        with torch.no_grad():
            x=torch.randn(1,3,5,9);old,_=local_bn_relu(x,bn);got,s=local_bn_relu(x,bn,old,torch.zeros(1,1,5,9,dtype=torch.bool))
            self.assertTrue(torch.equal(got,old));self.assertEqual(calls,[135]);self.assertEqual(s['processed_relu_elements'],0)
            reset,_=local_bn_relu(x+1,bn);self.assertEqual(calls,[135,135]);self.assertTrue(torch.equal(reset,torch.relu(bn(x+1))))
        handle.remove()

    def test_guards(self):
        bn=torch.nn.BatchNorm2d(2).eval();x=torch.zeros(1,2,4,4)
        with self.assertRaises(ValueError):local_bn_relu(x,bn)
        with torch.no_grad():
            with self.assertRaises(ValueError):local_bn_relu(x,bn,x,torch.zeros(4,4,dtype=torch.bool))
            with self.assertRaises(ValueError):local_bn_relu(x,bn.train())


if __name__=='__main__':unittest.main()

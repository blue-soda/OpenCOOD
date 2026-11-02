import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_integer_event_reference import dense_integer_reference
from opencood.models.sub_modules.int_radix_integer_reference import radix_integer_reference


class RadixIntegerTests(unittest.TestCase):
    def test_signed_mixed_digits_boundary_and_chunks_exact(self):
        torch.manual_seed(303)
        for kernel in (1,3):
            conv=nn.Conv2d(5,4,kernel,padding=kernel//2);values=torch.randint(-65536,65536,(1,5,4,5),dtype=torch.int64);weight=torch.randint(-32768,32768,conv.weight.shape,dtype=torch.int16)
            with torch.no_grad():
                reference,_=dense_integer_reference(conv,values,weight)
                for chunk in (1,17,256):
                    result,cost=radix_integer_reference(conv,values,weight,chunk);self.assertTrue(torch.equal(result,reference));self.assertLess(cost['partial_absolute_bound'],2**24)

    def test_zero_and_worst_partial_and_overflow_guard(self):
        conv=nn.Conv2d(256,1,1);weight=torch.full(conv.weight.shape,255,dtype=torch.int16)
        with torch.no_grad():
            result,cost=radix_integer_reference(conv,torch.full((1,256,1,1),255,dtype=torch.int64),weight);self.assertEqual(int(result),256*255*255);self.assertEqual(cost['partial_gemms'],1)
            zero,cost=radix_integer_reference(conv,torch.zeros(1,256,1,1,dtype=torch.int64),weight);self.assertEqual(int(zero),0);self.assertEqual(cost['fp32_reference_macs'],0)
            with self.assertRaises(ValueError):radix_integer_reference(conv,torch.full((1,256,1,1),2**50,dtype=torch.int64),weight)


if __name__=='__main__':unittest.main()

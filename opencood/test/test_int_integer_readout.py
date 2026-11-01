import unittest
import torch
from opencood.models.sub_modules.int_integer_readout import bounded_integer_conv,IntegerCodeConv


class IntegerReadoutTests(unittest.TestCase):
    def test_dense_gemm_matches_explicit_int64_spatial_reference(self):
        x=(torch.arange(2*3*4).reshape(1,2,3,4)%9).int()
        q=((torch.arange(3*2*3*3).reshape(3,2,3,3)%255)-127).to(torch.int8)
        actual,_=bounded_integer_conv(x,q,8,1)
        expected=torch.zeros_like(actual,dtype=torch.int64)
        for o in range(3):
            for y in range(3):
                for z in range(4):
                    for c in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy,ix=y+ky-1,z+kx-1
                                if 0<=iy<3 and 0<=ix<4:
                                    expected[0,o,y,z]+=int(x[0,c,iy,ix])*int(q[o,c,ky,kx])
        self.assertTrue(torch.equal(actual.long(),expected))

    def test_worst_case_bound_reached_without_overflow(self):
        x=torch.full((1,384,3,3),8,dtype=torch.int32)
        for sign in (-1,1):
            q=torch.full((1,384,3,3),127*sign,dtype=torch.int8)
            actual,_=bounded_integer_conv(x,q,8,0)
            self.assertEqual(int(actual[0,0,1,1]),sign*3511296)

    def test_integer_count_and_event_outputs_exact_with_analog_tail(self):
        torch.manual_seed(303)
        conv=torch.nn.Conv2d(2,3,3,padding=1)
        scales=torch.tensor([.3,1.7]).reshape(1,2,1,1)
        count=IntegerCodeConv(conv,scales,'folded_count').eval()
        event=IntegerCodeConv(conv,scales,'event8').eval()
        x=torch.randn(1,2,4,5)
        with torch.no_grad():a=count(x);b=event(x)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(count.quantization,event.quantization)
        self.assertEqual(event.last_stats['integer_time_block_additions'],7*a.numel())


if __name__=='__main__':unittest.main()

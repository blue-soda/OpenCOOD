import unittest
import torch
from opencood.models.sub_modules.int_integer_readout import bounded_integer_conv,IntegerCodeConv,bitplane_cost


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

    def test_bitplane_all_counts_negative_weights_and_tail(self):
        conv=torch.nn.Conv2d(1,2,3,padding=1)
        with torch.no_grad():conv.weight[0].fill_(-.7);conv.weight[1].fill_(.3)
        scales=torch.ones(1,1,1,1)
        count=IntegerCodeConv(conv,scales,'folded_count').eval()
        bits=IntegerCodeConv(conv,scales,'bitplane').eval()
        x=torch.arange(12).float().reshape(1,1,3,4)/8
        with torch.no_grad():a=count(x);b=bits(x)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(count.quantization,bits.quantization)
        self.assertEqual(bits.last_stats['integer_bit_shifts'],3*a.numel())

    def test_bitplane_delivery_histogram_matches_explicit_fanout(self):
        for h,w in ((1,1),(1,9),(3,3)):
            x=(torch.arange(h*w).reshape(1,1,h,w)%9).int()
            actual=bitplane_cost(x,2)
            expected=[0]*9
            for y in range(h):
                for z in range(w):
                    for dy in (-1,0,1):
                        for dz in (-1,0,1):
                            if 0<=y+dy<h and 0<=z+dz<w:expected[int(x[0,0,y,z])]+=2
            self.assertEqual(actual['valid_delivery_histogram'],expected)
            self.assertLessEqual(actual['bitplane_shifted_ac'],actual['unary_event_ac'])
            self.assertGreaterEqual(actual['bitplane_shifted_ac'],actual['sparse_count_mac'])


if __name__=='__main__':unittest.main()

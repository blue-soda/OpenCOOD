import unittest
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_folded_readout import FoldedCodeConv,quantize_weight_int8


class FoldedReadoutTests(unittest.TestCase):
    def test_int8_roundtrip_bounds_and_zero_channel(self):
        w=torch.tensor([[-2.,0.,1.],[0.,0.,0.]]).view(2,1,1,3)
        q,s,d=quantize_weight_int8(w)
        self.assertEqual(q.dtype,torch.int8)
        self.assertEqual(q.flatten().tolist(),[-127,0,64,0,0,0])
        self.assertEqual(float(s[1]),1.)
        self.assertTrue(bool(((d-w).abs()<=s/2+1e-6).all()))
        self.assertTrue(torch.equal(d[1],w[1]))

    def test_int8_same_banks_and_fp64_algebra_for_both_accumulators(self):
        torch.manual_seed(303)
        conv=torch.nn.Conv2d(2,3,3,padding=1).double()
        scales=torch.tensor([.3,1.7],dtype=torch.float64).view(1,2,1,1)
        count=FoldedCodeConv(conv,scales,'folded_count',8).eval()
        event=FoldedCodeConv(conv,scales,'event8',8).eval()
        self.assertEqual(count.quantization,event.quantization)
        for bank in ('event_int8','event_weight_scale','tail_int8','tail_weight_scale'):
            self.assertTrue(torch.equal(getattr(count,bank),getattr(event,bank)))
        x=torch.randn(2,2,4,5,dtype=torch.float64)
        with torch.no_grad():a=count(x);b=event(x)
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(conv.weight,count.source.weight))
        self.assertEqual(count.quantization['conceptual_two_bank_int8_bytes'],2*conv.weight.numel())

    def test_int8_rejects_nonfinite_weights(self):
        with self.assertRaises(ValueError):quantize_weight_int8(torch.full((1,1,3,3),float('nan')))

    def test_fp64_algebra_with_tail_negative_values_and_bias(self):
        torch.manual_seed(303)
        conv = torch.nn.Conv2d(2,3,3,padding=1,bias=True).double()
        scales = torch.tensor([.3,1.7],dtype=torch.float64).view(1,2,1,1)
        x = torch.randn(2,2,4,5,dtype=torch.float64)
        positive = x.clamp_min(0)
        tail = positive > scales
        drive = torch.floor(torch.minimum(positive,scales)/scales*65536).long()
        drive.masked_fill_(tail,0)
        counts = torch.floor_divide(drive*8+32768,65536)
        decoded = torch.where(tail,positive,counts.double()*(scales/8))
        reference = F.conv2d(decoded,conv.weight,conv.bias,padding=1)
        for mode,calls in [('folded_count',2),('event8',9)]:
            module = FoldedCodeConv(conv,scales,mode).eval()
            with torch.no_grad():actual = module(x)
            torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
            self.assertEqual(module.last_stats['functional_conv_flops'],calls*2*reference.numel()*2*9)
            self.assertEqual(sum(p.numel() for p in module.parameters()),sum(p.numel() for p in conv.parameters()))

    def test_zero_adds_bias_once_and_tail_emits_no_events(self):
        conv = torch.nn.Conv2d(1,1,3,padding=1)
        conv.bias.data.fill_(2.)
        for mode in ('folded_count','event8'):
            module = FoldedCodeConv(conv,torch.ones(1,1,1,1),mode).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(module(torch.zeros(1,1,3,4)),torch.full((1,1,3,4),2.)))
                x = torch.full((1,1,3,4),2.)
                torch.testing.assert_close(module(x),conv(x))
                self.assertEqual(module.last_stats['spike_count'],0)
                self.assertEqual(module.last_stats['analog_elements'],12)


if __name__ == '__main__':unittest.main()

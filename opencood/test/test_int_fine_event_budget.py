import unittest
import torch
from torch import nn
from opencood.tools.probe_int_fine_event_budget import input_fanout


class FineFanoutTests(unittest.TestCase):
    def test_strides_boundaries_and_current_sparse_deliveries(self):
        torch.manual_seed(3)
        for height,width in ((5,7),(6,8)):
            for kernel in (1,3):
                for stride in (1,2):
                    for padding in sorted({0,kernel//2}):
                        with self.subTest(shape=(height,width),kernel=kernel,stride=stride,padding=padding):
                            conv=nn.Conv2d(2,3,kernel,stride=stride,padding=padding,bias=False)
                            x=(torch.rand(1,2,height,width)>.5).float()
                            with torch.no_grad():conv.weight.fill_(1);out=conv(x)
                            fan=input_fanout(conv,x,out);expected=torch.zeros_like(fan)
                            for oy in range(out.shape[-2]):
                                for ox in range(out.shape[-1]):
                                    for ky in range(kernel):
                                        for kx in range(kernel):
                                            y=oy*stride-padding+ky;xx=ox*stride-padding+kx
                                            if 0<=y<height and 0<=xx<width:expected[0,0,y,xx]+=1
                            self.assertTrue(torch.equal(fan,expected))
                            self.assertEqual(float(out.sum()),float((x*fan).sum())*3)


if __name__=='__main__':unittest.main()

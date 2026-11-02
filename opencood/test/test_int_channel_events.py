import unittest
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_channel_events import channel_event_delta


class ChannelEventTests(unittest.TestCase):
    def test_signed_channel_events_match_dense_and_chunking(self):
        torch.manual_seed(303)
        for kernel in (1,3):
            conv=nn.Conv2d(3,4,kernel,padding=kernel//2).double();delta=torch.randn(1,3,4,5,dtype=torch.float64);delta[delta.abs()<1]=0
            with torch.no_grad():
                reference=F.conv2d(delta,conv.weight,None,padding=kernel//2)
                a,stats=channel_event_delta(conv,delta,2);b,other=channel_event_delta(conv,delta,10000)
                self.assertTrue(torch.allclose(a,reference,atol=1e-12,rtol=1e-12));self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
                expected=0
                for _,channel,y,x in torch.nonzero(delta,as_tuple=False).tolist():
                    expected+=sum(0<=y+kernel//2-ky<4 and 0<=x+kernel//2-kx<5 for ky in range(kernel) for kx in range(kernel))
                self.assertEqual(stats['scalar_deliveries'],expected);self.assertEqual(stats['executed_macs'],expected*4);self.assertEqual(other['executed_macs'],stats['executed_macs'])

    def test_zero_input_avoids_conv_and_rejects_training(self):
        conv=nn.Conv2d(2,3,3,padding=1);calls=[];handle=conv.register_forward_hook(lambda *_:calls.append(1));delta=torch.zeros(1,2,3,4)
        with torch.no_grad():
            result,stats=channel_event_delta(conv,delta);self.assertEqual(torch.count_nonzero(result),0);self.assertEqual(stats['executed_macs'],0);self.assertEqual(calls,[])
        handle.remove()
        with self.assertRaises(ValueError):channel_event_delta(conv,delta)


if __name__=='__main__':unittest.main()

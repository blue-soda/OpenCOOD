import unittest
import torch
from torch import nn

from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class NonzeroFusionTest(unittest.TestCase):
    @torch.no_grad()
    def test_zeros_edges_changes_and_counts(self):
        torch.manual_seed(303)
        conv=nn.Conv2d(384,128,3,padding=1).cuda().eval()
        layer=NonzeroFusionCache(conv)
        x=torch.zeros(1,384,8,9,device='cuda')
        prev=None
        for i in range(4):
            if i==1:x[0,300,0,0]=-1.25
            if i==3:x[0,370,4,4]=.75
            layer.begin(prev,None,False)
            expected=layer(x)
            static=layer.ledger['nonzero_taps']
            layer.begin(prev,None,True)
            actual=layer(x)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.allclose(actual,conv(x),atol=2e-5,rtol=1e-4))
            self.assertEqual(static,[0,4,4,13][i])
            self.assertEqual(layer.ledger['nonzero_taps'],[0,4,0,9][i])
            prev=dict(fusion_cache=layer.new)


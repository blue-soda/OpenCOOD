import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_exact_fusion_cache import ExactFusionCache


class FusionTest(unittest.TestCase):
    @torch.no_grad()
    def test_local_equals_dense_for_empty_and_clipped_changes(self):
        torch.manual_seed(303)
        conv=nn.Conv2d(384,128,3,padding=1)
        layer=ExactFusionCache(conv,block=16)
        x=torch.randn(1,384,8,9)
        prev=None
        for step in range(4):
            y=x.clone()
            y[:,:256]+=.1
            if step==2:y[:,256:,0,0]+=.5
            if step==3:y[:,256:,3,4]-=.75
            layer.begin(prev,None,False)
            reference=layer(y)
            layer.begin(prev,None,True)
            actual=layer(y)
            self.assertTrue(torch.equal(actual,reference))
            self.assertTrue(torch.allclose(actual,conv(y),atol=1e-5,rtol=1e-4))
            self.assertEqual(layer.ledger['dirty'],[72,0,4,9][step])
            prev=dict(fusion_cache=layer.new)
            x=y


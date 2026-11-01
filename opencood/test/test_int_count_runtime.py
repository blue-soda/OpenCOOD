import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_if_activation import FixedPointIFActivation,readout_coded_anchor
from opencood.models.sub_modules.int_count_runtime import CountOnlyActivation,runtime_count_anchor


class CountRuntimeTests(unittest.TestCase):
    def test_exact_codec_at_ties_tails_and_random_values(self):
        torch.manual_seed(303)
        ref=FixedPointIFActivation([.3,1.7],'count',8,True,32768).eval()
        fast=CountOnlyActivation(ref).eval()
        values=torch.cat([torch.randn(253)*2,torch.tensor([0.,.0625,1.])]).view(1,1,16,16)
        x=values.repeat(2,2,1,1)*ref.scales
        with torch.no_grad():self.assertTrue(torch.equal(ref(x),fast(x)))
        self.assertFalse(hasattr(fast,'last_stats'))

    def test_predictions_and_cache_exact_on_refresh_and_reuse(self):
        torch.manual_seed(303)
        base=fixture();scales=dict(input=torch.ones(384),middle=torch.ones(256))
        ref=readout_coded_anchor(base,scales,'count',8,True,32768)
        fast=runtime_count_anchor(base,scales)
        self.assertEqual(sum(p.numel() for p in ref.parameters()),sum(p.numel() for p in fast.parameters()))
        left=right=None
        with torch.no_grad():
            for index in range(4):
                x=torch.randn(1,64,24,32)
                meta=dict(timestamp_us=index*100000,segment='a',pose=np.eye(4))
                a,left,_=ref.step_spatial(x,left,meta)
                b,right,_=fast.step_spatial(x,right,meta)
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
                self.assertTrue(all(torch.equal(left[k],right[k]) for k in ('value','coarse')))
                self.assertTrue(torch.equal(left['anchor']['value'],right['anchor']['value']))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in fast.modules()))


if __name__=='__main__':unittest.main()

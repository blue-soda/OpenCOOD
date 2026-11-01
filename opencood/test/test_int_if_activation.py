import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_if_activation import FixedPointIFActivation,coded_anchor


class IFActivationTests(unittest.TestCase):
    def test_explicit_if_equals_integer_count_at_boundaries_and_random_inputs(self):
        torch.manual_seed(303)
        for steps in (4,8,16):
            x=torch.cat([torch.linspace(-.5,2.,127),torch.rand(137)*2,torch.tensor([0.,1./steps,1.,2.])]).reshape(1,1,1,-1).repeat(1,2,3,1)
            for scales in ([1.,1.],[.3,2.]):
                count=FixedPointIFActivation(scales,'count',steps).eval();spike=FixedPointIFActivation(scales,'if',steps).eval()
                with torch.no_grad():a=count(x);b=spike(x)
                self.assertTrue(torch.equal(a,b));self.assertEqual(count.last_stats['spike_count'],spike.last_stats['spike_count'])

    def test_clip_and_membrane_reset(self):
        clip=FixedPointIFActivation([1.],'clip').eval();spike=FixedPointIFActivation([1.],'if').eval()
        x=torch.tensor([-.2,0.,.2,1.,2.]).reshape(1,1,1,-1)
        with torch.no_grad():
            self.assertTrue(torch.equal(clip(x),x.clamp(0,1)))
            a=spike(x);spike(x*3);b=spike(x)
            self.assertTrue(torch.equal(a,b));self.assertEqual(spike.last_stats['persistent_membrane_bytes'],0)
            self.assertEqual(float(spike(-torch.ones_like(x)).sum()),0.)

    def test_shared_base_is_unchanged_and_reuse_skips_codec(self):
        base=fixture();model=coded_anchor(base,torch.ones(256),'if',8);codec=model.core.deep_update.residual[1]
        self.assertIsInstance(base.core.deep_update.residual[1],torch.nn.ReLU)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta);calls=codec.calls
            _,new,info=model.step_spatial(x,state,dict(meta,timestamp_us=100000))
            self.assertIsNone(info['late_refresh']);self.assertEqual(codec.calls,calls)
            self.assertIs(new['anchor']['value'],state['anchor']['value'])

    def test_invalid_configuration(self):
        for scale in ([0.],[-1.],[float('nan')]):
            with self.assertRaises(ValueError):FixedPointIFActivation(scale)
        with self.assertRaises(ValueError):FixedPointIFActivation([1.],steps=0)


if __name__=='__main__':unittest.main()

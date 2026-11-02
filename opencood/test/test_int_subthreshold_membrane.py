import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector,quantize_ste


class SubthresholdTests(unittest.TestCase):
    def test_ste_forward_and_gradient(self):
        x=torch.tensor([-.38,-.12,.13,.49],requires_grad=True);q=quantize_ste(x,.25)
        self.assertTrue(torch.equal(q.detach(),torch.tensor([-.5,0.,.25,.5])))
        q.sum().backward();self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))
        with self.assertRaises(ValueError):quantize_ste(x,0)

    def test_continuous_matches_original_through_refresh_and_warp(self):
        source=ProtectedMultiscaleRefreshDetector(fixture());model=SubthresholdMembraneDetector(source,'continuous');a=b=None
        with torch.no_grad():
            for i in range(6):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.31;meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                p,a,ia=source.step_spatial(x,a,meta);q,b,ib=model.step_spatial(x,b,meta)
                for k in p:self.assertTrue(torch.equal(p[k],q[k]))
                for k in source.state_keys:self.assertTrue(torch.equal(a[k],b[k]))
                self.assertEqual(ia['cache_bytes'],ib['cache_bytes'])

    def test_residual_is_task_state_not_extra_tensor(self):
        source=ProtectedMultiscaleRefreshDetector(fixture());carry=SubthresholdMembraneDetector(source,'carry');drop=SubthresholdMembraneDetector(source,'drop');meta=dict(timestamp_us=0,segment='a',pose=np.eye(4));x=torch.randn(1,64,24,32)
        with torch.no_grad():
            p,a,ia=carry.step_spatial(x,None,meta);q,b,ib=drop.step_spatial(x,None,meta)
            for k in p:self.assertTrue(torch.equal(p[k],q[k]))
            self.assertFalse(torch.equal(a['coarse'],b['coarse']))
            self.assertTrue(torch.equal(quantize_ste(a['coarse'],.25),b['coarse']))
            self.assertEqual(ia['cache_bytes'],ib['cache_bytes'])
            self.assertEqual({k for k,v in a.items() if torch.is_tensor(v)},set(source.state_keys))
            self.assertGreater(ia['membrane']['residual_square_sum'],0)

    def test_state_reset_and_surrogate_backward(self):
        source=ProtectedMultiscaleRefreshDetector(fixture());model=SubthresholdMembraneDetector(source,'carry');x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta);later=dict(meta,timestamp_us=100000,segment='b')
            p,a,_=model.step_spatial(x,state,later);q,b,_=model.step_spatial(x,None,later)
            for k in p:self.assertTrue(torch.equal(p[k],q[k]))
            for k in source.state_keys:self.assertTrue(torch.equal(a[k],b[k]))
            with self.assertRaises(ValueError):model.step_spatial(x,state,meta)
        model.zero_grad(set_to_none=True);p,_,_=model.step_spatial(x,None,meta,False);p['psm'].square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()

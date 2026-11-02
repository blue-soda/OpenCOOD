import unittest
import torch
import numpy as np
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector,MultiscaleRefreshDetector


class MultiscaleRefreshTests(unittest.TestCase):
    def test_original_period1_preserves_dense_and_skips_light(self):
        base=fixture();model=MultiscaleRefreshDetector(base,'original',period=1)
        calls=[];handles=[m.register_forward_hook(lambda *_:calls.append(1)) for m in (model.core.coarse_light,model.core.deep_light)]
        a=b=None
        with torch.no_grad():
            for i in range(3):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.1
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                expected,a,_=base.step_spatial(x,a,meta);actual,b,_=model.step_spatial(x,b,meta)
                for k in expected:self.assertTrue(torch.equal(expected[k],actual[k]))
        for h in handles:h.remove()
        self.assertEqual(calls,[])

    def test_light_really_skips_two_encoders_and_preserves_reset_guards(self):
        model=MultiscaleRefreshDetector(fixture(),'gate');state=None;calls=[0,0]
        def hit(i):
            def count(*_):calls[i]+=1
            return count
        handles=[m.register_forward_hook(hit(i)) for i,m in enumerate((model.core.coarse_encoder,model.core.deep_encoder))]
        x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for i in range(5):
                _,state,info=model.step_spatial(x,state,dict(timestamp_us=i*100000,segment='a',pose=np.eye(4)))
                self.assertEqual(info['full_refresh'],'initial' if i==0 else 'period' if i==4 else None)
                self.assertEqual(calls,[1+int(i==4)]*2)
            for reason,meta in (('gap',dict(timestamp_us=700001,segment='a',pose=np.eye(4))),('scene',dict(timestamp_us=500000,segment='b',pose=np.eye(4)))):
                _,_,info=model.step_spatial(x,state,meta);self.assertEqual(info['reset'],reason)
            with self.assertRaises(ValueError):model.step_spatial(x,state,dict(timestamp_us=400000,segment='a',pose=np.eye(4)))
        for h in handles:h.remove()

    def test_coarse_only_control_and_real_delta_t(self):
        base=fixture();model=TemporalPyramidDetector(base,'original');a=b=None
        with torch.no_grad():
            for stamp in (0,70000,240000):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=stamp,segment='a',pose=np.eye(4))
                pa,a,_=base.step_spatial(x,a,meta);pb,b,_=model.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa))
            self.assertAlmostEqual(model.core.coarse_update.delta_t,.17)

    def test_own_history_tbptt_light_gradient(self):
        model=MultiscaleRefreshDetector(fixture(),'gate');state=None
        for p in model.parameters():p.requires_grad_(False)
        for name in ('coarse_light','deep_light','coarse_update','deep_update'):
            for p in getattr(model.core,name).parameters():p.requires_grad_(True)
        for i in range(3):
            pred,state,_=model.step_spatial(torch.randn(1,64,24,32),state,
                dict(timestamp_us=i*100000,segment='a',pose=np.eye(4)),detach_state=False)
        pred['psm'].square().mean().backward()
        for name in ('coarse_light','deep_light'):
            gradients=[p.grad for p in getattr(model.core,name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        self.assertIsNotNone(state['coarse'].grad_fn)


if __name__=='__main__':unittest.main()

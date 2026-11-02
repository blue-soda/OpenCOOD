import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector


class ProtectedRefreshTests(unittest.TestCase):
    def test_period_one_exact_even_after_ordinary_weights_change(self):
        base=fixture();m=ProtectedMultiscaleRefreshDetector(base,period=1)
        with torch.no_grad():
            for module in (m.core.coarse_update,m.core.deep_update,m.core.coarse_light,m.core.deep_light):
                for p in module.parameters():p.add_(.1)
            a=b=None
            for i in range(3):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=i*100000,segment='a',pose=np.eye(4))
                pa,a,_=base.step_spatial(x,a,meta);pb,b,_=m.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa))

    def test_dense_observation_control_has_same_refresh_schedule(self):
        base=fixture();a=ProtectedMultiscaleRefreshDetector(base);b=ProtectedMultiscaleRefreshDetector(base,False)
        b.load_state_dict(a.state_dict());sa=sb=None
        with torch.no_grad():
            for i in range(5):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=i*100000,segment='a',pose=np.eye(4));ca=[];cb=[]
                ha=a.core.deep_encoder.register_forward_hook(lambda *_:ca.append(1));hb=b.core.deep_encoder.register_forward_hook(lambda *_:cb.append(1))
                try:
                    _,sa,ia=a.step_spatial(x,sa,meta);_,sb,ib=b.step_spatial(x,sb,meta)
                finally:ha.remove();hb.remove()
                self.assertEqual(ia['full_refresh'],ib['full_refresh']);self.assertEqual(len(ca),int(bool(ia['full_refresh'])));self.assertEqual(cb,[1])

    def test_refresh_preserves_student_history_not_parent_history(self):
        m=ProtectedMultiscaleRefreshDetector(fixture(),period=2);x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,s,_=m.step_spatial(x,None,meta);_,s,_=m.step_spatial(x,s,dict(meta,timestamp_us=100000))
            altered=dict(s,coarse=s['coarse']+2,deep=s['deep']+2)
            _,a,info=m.step_spatial(x,s,dict(meta,timestamp_us=200000));_,b,_=m.step_spatial(x,altered,dict(meta,timestamp_us=200000))
            self.assertEqual(info['full_refresh'],'period');self.assertFalse(torch.equal(a['coarse'],b['coarse']))

    def test_frozen_refresh_gradients_and_learned_light(self):
        m=ProtectedMultiscaleRefreshDetector(fixture());s=None
        for n,p in m.named_parameters():p.requires_grad_(n.startswith(('core.coarse_update.','core.deep_update.','core.coarse_light.','core.deep_light.')))
        for i in range(4):
            pred,s,_=m.step_spatial(torch.randn(1,64,24,32),s,dict(timestamp_us=i*100000,segment='a',pose=np.eye(4)),detach_state=False)
        pred['psm'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in m.core.refresh_updates.parameters()))
        for module in (m.core.coarse_light,m.core.deep_light,m.core.coarse_update,m.core.deep_update):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)


if __name__=='__main__':unittest.main()

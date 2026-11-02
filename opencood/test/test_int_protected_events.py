import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_protected_events import ProtectedEventDetector


class ProtectedEventTests(unittest.TestCase):
    def test_continuous_exact_and_quantized_sparse_matches_dense(self):
        source=ProtectedMultiscaleRefreshDetector(fixture());models=[source]+[ProtectedEventDetector(source,m) for m in ('continuous','quant_dense','event_phase')];states=[None]*4;calls=[]
        h=models[-1].core.deblocks[1][0].register_forward_hook(lambda *_:calls.append(1))
        with torch.no_grad():
            for i in range(6):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.41;meta=dict(timestamp_us=i*100000,segment='a',pose=pose);preds=[]
                for j,m in enumerate(models):
                    p,states[j],info=m.step_spatial(x,states[j],meta);preds.append(p)
                    self.assertEqual(info['cache_bytes'],sum(v.numel()*v.element_size() for v in states[j].values() if torch.is_tensor(v)))
                for k in preds[0]:self.assertTrue(torch.equal(preds[0][k],preds[1][k]));self.assertTrue(torch.allclose(preds[2][k],preds[3][k],atol=2e-5,rtol=1e-4))
                for k in source.state_keys:self.assertTrue(torch.equal(states[0][k],states[1][k]));self.assertTrue(torch.equal(states[2][k],states[3][k]))
                self.assertTrue(torch.equal(states[0]['coarse'],states[2]['coarse']))
            self.assertEqual(len(calls),2)
        h.remove()

    def test_all_extra_states_reset_and_time_guard(self):
        model=ProtectedEventDetector(ProtectedMultiscaleRefreshDetector(fixture()),'event_phase');x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,s,_=model.step_spatial(x,None,meta)
            for extra in (dict(timestamp_us=100000,segment='b'),dict(timestamp_us=300001),dict(timestamp_us=100000,reset_before=True)):
                m=dict(meta,**extra);p,new,_=model.step_spatial(x,s,m);q,fresh,_=model.step_spatial(x,None,m)
                for k in p:self.assertTrue(torch.equal(p[k],q[k]))
                for k in ('value','coarse','deep','sent','projection'):self.assertTrue(torch.equal(new[k],fresh[k]))
            with self.assertRaises(ValueError):model.step_spatial(x,s,meta)


if __name__=='__main__':unittest.main()

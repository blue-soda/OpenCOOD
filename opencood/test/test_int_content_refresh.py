import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_content_refresh import ContentRefreshDetector,replay_content,decide_refresh


class ContentRefreshTests(unittest.TestCase):
    def test_period_control_preserves_source_and_actual_heavy_calls(self):
        source=ProtectedMultiscaleRefreshDetector(fixture());m=ContentRefreshDetector(source);a=b=None;calls=[];h=m.core.coarse_encoder.register_forward_hook(lambda *_:calls.append(1))
        with torch.no_grad():
            for i in range(7):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=i*100000,segment='a',pose=np.eye(4));p,a,ia=source.step_spatial(x,a,meta);q,b,ib=m.step_spatial(x,b,meta)
                self.assertEqual(ia['full_refresh'],ib['full_refresh']);self.assertTrue(all(torch.equal(p[k],q[k]) for k in p));self.assertTrue(all(torch.equal(a[k],b[k]) for k in source.state_keys));self.assertEqual(len(calls),1+i//4)
        h.remove()

    def test_content_replay_and_max_age_causality(self):
        rows=[dict(timestamp_us=i*100000,segment='a',evidence=.8 if i==2 else .1) for i in range(10)]
        self.assertEqual(replay_content(rows,'period4'),['initial',None,None,None,'period',None,None,None,'period',None])
        self.assertEqual(replay_content(rows,'content',.5),['initial',None,'content',None,None,None,None,None,'period',None])
        with self.assertRaises(ValueError):replay_content([rows[0],rows[0]],'content',.5)
        state=dict(scene='a',timestamp_us=600000,refresh_timestamp_us=0,since_refresh=3)
        self.assertEqual(decide_refresh(dict(timestamp_us=650000,segment='a'),state,0.,'content',.5),'age')
        self.assertEqual(decide_refresh(dict(timestamp_us=900001,segment='a'),state,0.,'content',.5),'gap')


if __name__=='__main__':unittest.main()

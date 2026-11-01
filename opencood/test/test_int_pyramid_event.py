import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_pyramid_event import EvidenceTraceAnchor,EventAnchoredPyramidDetector
from opencood.models.sub_modules.int_event_refresh import replay_refresh_trace


class PyramidEventTests(unittest.TestCase):
    def test_evidence_observer_preserves_strict_predictions(self):
        base=fixture();plain=AnchoredLatePyramidDetector(base);trace=EvidenceTraceAnchor(base)
        a=b=None
        with torch.no_grad():
            for i in range(4):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.2
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                p,a,pi=plain.step_spatial(x,a,meta);q,b,qi=trace.step_spatial(x,b,meta)
                for key in p:self.assertTrue(torch.equal(p[key],q[key]))
                self.assertEqual(pi['late_refresh'],qi['late_refresh'])
                self.assertGreaterEqual(qi['refresh_evidence'],0.)

    def test_no_evidence_trigger_skips_deep_until_own_age_or_coverage(self):
        base=fixture();model=EventAnchoredPyramidDetector(base,'event',1e9);calls=[]
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta);anchor=state['anchor']['value']
            hook=base.core.deep_encoder.register_forward_hook(lambda *_:calls.append(1))
            try:
                for stamp in (100000,200000):
                    _,state,info=model.step_spatial(x,state,dict(meta,timestamp_us=stamp))
                    self.assertIsNone(info['late_refresh']);self.assertIs(state['anchor']['value'],anchor)
                self.assertFalse(calls)
                _,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=250000))
                self.assertEqual(info['late_refresh'],'age');self.assertEqual(len(calls),1)
                pose=np.eye(4);pose[0,3]=2.
                _,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=240000,pose=pose))
                self.assertEqual(info['late_refresh'],'coverage')
            finally:hook.remove()

    def test_trace_uses_candidate_anchor_and_counts_forced_work(self):
        rows=[dict(timestamp_us=i*100000,segment='a',reset='initial' if i==0 else None,evidence=2.,
                   coverage_by_anchor={str(j):float(i==2 and j==0) for j in range(i)}) for i in range(4)]
        slow=replay_refresh_trace(rows,'threshold',100.)
        fast=replay_refresh_trace(rows,'threshold',.1)
        self.assertEqual(slow[2]['refresh'],'coverage')
        self.assertEqual(fast[2]['refresh'],'evidence')
        self.assertEqual(fast[1]['anchor_index'],1)


if __name__=='__main__':unittest.main()

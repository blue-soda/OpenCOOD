import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_pyramid_anchor_patch import PatchedLatePyramidDetector


class LongRefreshTests(unittest.TestCase):
    def test_periods_and_original_anchor_identity(self):
        base=fixture();x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for period in (2,3,4,6):
                model=AnchoredLatePyramidDetector(base,period=period,max_age=.65)
                state=None
                for i in range(2*period+1):
                    previous=state
                    _,state,info=model.step_spatial(x,state,dict(timestamp_us=i*100000,segment='a',pose=np.eye(4)))
                    self.assertEqual(info['late_refresh'],'initial' if i==0 else ('period' if i%period==0 else None))
                    if i and i%period:
                        self.assertIs(state['anchor']['value'],previous['anchor']['value'])
                        self.assertEqual(state['anchor']['timestamp_us'],previous['anchor']['timestamp_us'])

    def test_anchor_age_independent_of_gap(self):
        model=AnchoredLatePyramidDetector(fixture(),period=6,max_age=.25)
        state=None;x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for i in range(4):
                _,state,info=model.step_spatial(x,state,dict(timestamp_us=i*100000,segment='a',pose=np.eye(4)))
            self.assertEqual(info['late_refresh'],'age')
            self.assertIsNone(info['reset'])
        model.max_age=.65
        with torch.no_grad():
            _,_,info=model.step_spatial(x,state,dict(timestamp_us=600001,segment='a',pose=np.eye(4)))
        self.assertEqual(info['reset'],'gap')

    def test_long_period_scene_explicit_and_noncausal_guards(self):
        model=AnchoredLatePyramidDetector(fixture(),period=6,max_age=.65)
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta)
            for reason,extra in (('scene',dict(segment='b')),('explicit',dict(reset_before=True))):
                _,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=100000,**extra))
                self.assertEqual(info['late_refresh'],reason)
            with self.assertRaises(ValueError):model.step_spatial(x,state,meta)

    def test_long_patch_same_mask_dense_parity(self):
        base=fixture();local=PatchedLatePyramidDetector(base,period=4,max_age=.45)
        dense=PatchedLatePyramidDetector(base,period=4,max_age=.45,dense_reference=True)
        x=torch.randn(1,64,24,32);a=b=None
        with torch.no_grad():
            for i in range(5):
                pose=np.eye(4);pose[0,3]=i*.5
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                pa,a,ia=local.step_spatial(x,a,meta);pb,b,ib=dense.step_spatial(x,b,meta)
                for key in pa:self.assertTrue(torch.allclose(pa[key],pb[key],atol=1e-4,rtol=1e-4))
                self.assertEqual(ia['late_refresh'],ib['late_refresh'])
                self.assertTrue(torch.equal(a['anchor']['value'],b['anchor']['value']))

    def test_bounds(self):
        base=fixture()
        for period,age in ((0,.25),(7,.65),(True,.25),(4,0),(4,.66),(4,float('nan'))):
            with self.assertRaises(ValueError):AnchoredLatePyramidDetector(base,period=period,max_age=age)


if __name__=='__main__':unittest.main()

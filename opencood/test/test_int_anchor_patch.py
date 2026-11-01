import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_pyramid_anchor_patch import PatchedLatePyramidDetector


class AnchorPatchTests(unittest.TestCase):
    def test_summary_schema_stable_across_refresh_patch_and_empty_reuse(self):
        from opencood.tools.int_stage_summary import stage_means
        local=PatchedLatePyramidDetector(fixture());state=None;records=[]
        x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for i,shift in enumerate((0.,2.,2.,2.)):
                pose=np.eye(4);pose[0,3]=shift
                _,state,info=local.step_spatial(x,state,dict(timestamp_us=i*100000,segment='a',pose=pose))
                records.append(info)
        self.assertEqual(records[0]['late_refresh'],'initial')
        self.assertTrue(records[1]['repair']['active'])
        self.assertEqual(records[2]['late_refresh'],'period')
        self.assertFalse(records[3]['repair']['active'])
        for record in records:
            for stage in record['cache_stages']:
                self.assertEqual(set(stage),{'actual_write_fraction','executed_input_area_ratio'})
        self.assertEqual(len(stage_means(records)),3)

    def test_patch_is_transient_and_matches_same_mask_reference(self):
        base=fixture();local=PatchedLatePyramidDetector(base)
        reference=PatchedLatePyramidDetector(base,dense_reference=True)
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=local.step_spatial(x,None,meta)
            pose=np.eye(4);pose[0,3]=2.
            current=dict(meta,timestamp_us=100000,pose=pose)
            a,sa,ia=local.step_spatial(x,state,current)
            b,sb,ib=reference.step_spatial(x,state,current)
            self.assertTrue(ia['repair']['active']);self.assertIsNone(ia['late_refresh'])
            for k in a:self.assertTrue(torch.allclose(a[k],b[k],atol=1e-4,rtol=1e-4))
            self.assertEqual(ia['repair']['actual_write_fraction'],ib['repair']['actual_write_fraction'])
            self.assertIs(sa['anchor']['value'],state['anchor']['value'])
            self.assertEqual(sa['anchor']['timestamp_us'],0)
            self.assertTrue(np.array_equal(sa['anchor']['pose'],meta['pose']))
            _,_,next_info=local.step_spatial(x,sa,dict(current,timestamp_us=200000))
            self.assertEqual(next_info['late_refresh'],'period')

    def test_no_missing_cells_execute_no_deep_module(self):
        base=fixture();local=PatchedLatePyramidDetector(base);calls=[]
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=local.step_spatial(x,None,meta)
            h=base.core.deep_encoder.register_forward_hook(lambda *_:calls.append(1))
            try:_,_,info=local.step_spatial(x,state,dict(meta,timestamp_us=100000))
            finally:h.remove()
            self.assertEqual(calls,[]);self.assertFalse(info['repair']['active'])

    def test_odd_grid_border_crops_preserve_covered_cells(self):
        local=PatchedLatePyramidDetector(fixture())
        x=torch.randn(1,256,65,81);old=torch.randn(1,512,33,41)
        valid=torch.ones_like(old[:,:1],dtype=torch.bool);valid[...,0,:]=False;valid[...,:,-1]=False
        with torch.no_grad():
            actual,info=local.reuse_deep(x,old,valid)
            packed=local.patch_stage.pack(x,old,valid)
            expected=torch.where(valid,old,local.patch_stage(packed))
            self.assertTrue(torch.allclose(actual,expected,atol=1e-4,rtol=1e-4))
            self.assertTrue(torch.equal(actual[valid.expand_as(actual)],old[valid.expand_as(old)]))


if __name__=='__main__':unittest.main()

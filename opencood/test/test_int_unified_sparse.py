import unittest
from types import SimpleNamespace
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVCoreV2
from opencood.models.sub_modules.int_unified_sparse import SelectiveUnifiedBEV
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class SelectiveUnifiedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.core=UnifiedBEVCoreV2().eval()
        self.student=SimpleNamespace(core=self.core,training=False,memory_policy='recurrent',
                                     bounds=[-18.,-16.,-3.,18.,16.,3.])
        self.meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4),reset_before=False)

    def test_full_integration_matches_original_and_causal_resets(self):
        wrapper=SelectiveUnifiedBEV(self.student,0.,tile=4,execution='full')
        state=plain=None
        with torch.no_grad():
            for index in range(3):
                meta=dict(self.meta,timestamp_us=100000*(index+1));meta['pose']=np.eye(4);meta['pose'][0,3]=index
                x=torch.randn(1,64,32,40)
                expected,plain,_=self.core.step(x,plain,meta,self.student.bounds)
                actual,state,_=wrapper.step_spatial(x,state,meta)
                for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]))
            with self.assertRaises(ValueError):wrapper.step_spatial(x,state,meta)
            meta=dict(meta,segment='new')
            actual,_,info=wrapper.step_spatial(x,state,meta)
            expected,_,_=wrapper.step_spatial(x,None,meta)
            self.assertEqual(info['reset'],'scene')
            self.assertTrue(torch.equal(actual['rm'],expected['rm']))

    def test_reuse_skips_module_keeps_reference_and_age_forces_update(self):
        wrapper=SelectiveUnifiedBEV(self.student,1e6,tile=4,max_age=.2)
        x=torch.randn(1,64,32,40);calls=[]
        hook=self.core.update.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():
            _,first,_=wrapper.step_spatial(x,None,self.meta);calls.clear()
            meta=dict(self.meta,timestamp_us=200000)
            _,second,info=wrapper.step_spatial(x+.01,first,meta)
            self.assertEqual(calls,[])
            self.assertEqual(info['reused_fraction'],1.)
            self.assertTrue(torch.equal(first['value'],second['value']))
            self.assertTrue(torch.equal(first['reference'],second['reference']))
            _,third,info=wrapper.step_spatial(x+.02,second,dict(meta,timestamp_us=300000))
            self.assertGreater(len(calls),0)
            self.assertEqual(info['reused_fraction'],0.)
            self.assertEqual(float(third['age'].max()),0.)
        hook.remove()

    def test_sparse_and_masked_writeback_agree_with_new_view(self):
        sparse=SelectiveUnifiedBEV(self.student,1e6,tile=4,fallback_fraction=.95)
        masked=SelectiveUnifiedBEV(self.student,1e6,tile=4,execution='masked_dense',fallback_fraction=.95)
        a=b=None;partial=False
        with torch.no_grad():
            for index in range(4):
                meta=dict(self.meta,timestamp_us=100000*(index+1));meta['pose']=np.eye(4);meta['pose'][0,3]=index
                x=torch.randn(1,64,64,72)
                pa,a,ia=sparse.step_spatial(x,a,meta);pb,b,ib=masked.step_spatial(x,b,meta)
                self.assertEqual(ia['update_stats']['actual_write_fraction'],ib['update_stats']['actual_write_fraction'])
                for key in pa:self.assertTrue(torch.allclose(pa[key],pb[key],atol=2e-5,rtol=2e-5))
                self.assertTrue(torch.equal(a['reference'],b['reference']))
                partial=partial or 0<ia['reused_fraction']<1
            self.assertTrue(partial)

    def test_fixed_tile_batch_and_merged_plan_match_without_fallback(self):
        x=torch.randn(1,513,19,27);x[:,-1:]=1;old=x[:,256:512]
        mask=torch.zeros_like(old[:,:1],dtype=torch.bool)
        mask[...,0,0]=True;mask[...,8,8]=True;mask[...,8,12]=True;mask[...,-1,-1]=True
        with torch.no_grad():
            merged,wm,_=refresh_tiles(self.core.update,x,old,mask,1,2,4,allow_fallback=False)
            fixed,wf,_=refresh_tiles(self.core.update,x,old,mask,1,2,4,allow_fallback=False,layout='fixed',batch_size=32)
        self.assertTrue(torch.equal(wm,wf))
        self.assertTrue(torch.allclose(merged,fixed,atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.equal(fixed[~wf.expand_as(fixed)],old[~wf.expand_as(old)]))


if __name__=='__main__':unittest.main()

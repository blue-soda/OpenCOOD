import unittest
from types import SimpleNamespace
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVCoreV2,ResidualStateUpdate
from opencood.models.sub_modules.int_unified_gather import gathered_update
from opencood.models.sub_modules.int_cached_update_parts import CachedUpdateParts


class CachedPartsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(304)
        self.core=UnifiedBEVCoreV2().eval()
        self.student=SimpleNamespace(core=self.core,training=False,memory_policy='recurrent',bounds=[-18.,-16.,-3.,18.,16.,3.])
        self.meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4),reset_before=False)

    def test_period_one_matches_dense_sequence(self):
        wrapper=CachedUpdateParts(self.student,period=1);a=b=None
        with torch.no_grad():
            for i in range(3):
                meta=dict(self.meta,timestamp_us=(i+1)*100000);meta['pose']=np.eye(4);meta['pose'][0,3]=i
                x=torch.randn(1,64,32,40)
                pa,a,_=wrapper.step_spatial(x,a,meta);pb,b,_=self.core.step(x,b,meta,self.student.bounds)
                self.assertTrue(torch.equal(a['value'],b['value']))
                for k in pa:self.assertTrue(torch.equal(pa[k],pb[k]))

    def test_skip_still_updates_current_value_without_heavy_net(self):
        wrapper=CachedUpdateParts(self.student,period=3);calls=[]
        hook=self.core.update.net.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():
            x=torch.randn(1,64,32,40);_,first,_=wrapper.step_spatial(x,None,self.meta);calls.clear()
            _,second,info=wrapper.step_spatial(x+.1,first,dict(self.meta,timestamp_us=200000))
            self.assertEqual(calls,[]);self.assertEqual(info['parts_write_fraction'],0.)
            self.assertTrue(torch.equal(first['parts'],second['parts']))
            self.assertFalse(torch.equal(first['value'],second['value']))
            with self.assertRaises(ValueError):wrapper.step_spatial(x,second,dict(self.meta,timestamp_us=200000))
        hook.remove()

    def test_new_view_age_and_masked_equivalence(self):
        a=b=None;partial=False;age_refresh=False
        fast=CachedUpdateParts(self.student,period=20,tile=4,max_age=.2)
        dense=CachedUpdateParts(self.student,period=20,tile=4,max_age=.2,repair='masked_dense')
        with torch.no_grad():
            for i in range(5):
                meta=dict(self.meta,timestamp_us=(i+1)*100000);meta['pose']=np.eye(4);meta['pose'][0,3]=i
                x=torch.randn(1,64,64,72)
                pa,a,ia=fast.step_spatial(x,a,meta);pb,b,ib=dense.step_spatial(x,b,meta)
                self.assertEqual(ia['parts_write_fraction'],ib['parts_write_fraction'])
                self.assertLess(ia['max_parts_age'],.200001)
                for k in pa:self.assertTrue(torch.allclose(pa[k],pb[k],atol=3e-5,rtol=3e-5))
                partial=partial or 0<ia['parts_write_fraction']<1
                age_refresh=age_refresh or ia['parts_fallback']
            self.assertTrue(partial);self.assertTrue(age_refresh)
            _,_,info=fast.step_spatial(x,a,dict(meta,segment='new'))
            self.assertEqual(info['reset'],'scene');self.assertTrue(info['parts_anchor'])

    def test_gathered_parts_match_dense_net_at_edges(self):
        module=ResidualStateUpdate(evidence=6,channels=8).eval().double()
        with torch.no_grad():
            for p in module.parameters():p.uniform_(-.4,.4)
            x=torch.randn(1,15,11,13,dtype=torch.double);mask=torch.rand(1,1,11,13)>.7
            cache=torch.randn(1,16,11,13,dtype=torch.double)
            got=gathered_update(module,x,mask,cached_parts=cache)
            ref=torch.where(mask,module.net(x),cache)
            self.assertTrue(torch.allclose(got,ref,atol=1e-10,rtol=1e-10))


if __name__=='__main__':unittest.main()

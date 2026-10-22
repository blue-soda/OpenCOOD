import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_selected_memory import SelectedProjectedMemory, GATE_NAMES


class SelectedMemoryTests(unittest.TestCase):
    def source(self):
        torch.manual_seed(303)
        return projected_gru(INTFeatureMemory('gru',64,[-5,-4,-2,5,4,2]).eval(),32)[0].eval()

    def test_dense_period1_matches_original_and_resets(self):
        source=self.source();wrapper=SelectedProjectedMemory(source,period=1)
        left=right=None
        with torch.no_grad():
            for frame in range(5):
                pose=np.eye(4);pose[0,3]=frame*.1
                meta=dict(segment='a' if frame<3 else 'b',timestamp_us=frame*100000,pose=pose)
                x=torch.randn(1,64,7,11)
                a,left,_=source.step(x,left,meta);b,right,info=wrapper.step(x,right,meta)
                self.assertTrue(torch.equal(a,b));self.assertTrue(info['gate_refresh'])
                self.assertTrue(torch.equal(left['value'],right['value']))

    def test_recursive_selected_matches_same_mask_dense_and_keeps_owners(self):
        source=self.source()
        left=SelectedProjectedMemory(source,threshold=.5,period=3,execution='selected')
        right=SelectedProjectedMemory(source,threshold=.5,period=3,execution='masked_dense')
        ls=rs=None;refreshes=[]
        with torch.no_grad():
            for frame in range(8):
                pose=np.eye(4);pose[0,3]=frame*.2
                meta=dict(segment='a' if frame<5 else 'b',timestamp_us=frame*100000,pose=pose)
                x=torch.randn(1,64,7,11)
                previous=None if ls is None else ls['value'].clone()
                a,new_ls,ia=left.step(x,ls,meta);b,rs,ib=right.step(x,rs,meta)
                self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=1e-5))
                self.assertEqual(ia['gate_selected_fraction'],ib['gate_selected_fraction'])
                if previous is not None:self.assertTrue(torch.equal(previous,ls['value']))
                self.assertTrue(all(getattr(left.memory.fusion,n).context is None for n in GATE_NAMES))
                ls=new_ls;refreshes.append(ia['gate_refresh'])
        self.assertEqual(refreshes,[True,False,False,True,False,True,False,False])

    def test_retained_reference_does_not_follow_unselected_current(self):
        wrapper=SelectedProjectedMemory(self.source(),threshold=1e9,period=3)
        meta=dict(segment='a',timestamp_us=0,pose=np.eye(4))
        with torch.no_grad():
            _,old,_=wrapper.step(torch.randn(1,64,8,12),None,meta)
            meta=dict(meta,timestamp_us=100000)
            _,new,info=wrapper.step(torch.randn(1,64,8,12),old,meta)
        self.assertEqual(info['gate_selected_fraction'],0.)
        self.assertEqual(info['gate_extra_linear_macs'],0)
        for name in GATE_NAMES:
            for key in ('reference','response'):
                self.assertTrue(torch.equal(new['gate_cache']['gates'][name][key],old['gate_cache']['gates'][name][key]))


if __name__=='__main__':unittest.main()

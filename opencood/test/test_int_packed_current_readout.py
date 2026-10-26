import unittest
import numpy as np
import torch
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_local_current_readout import LocalCurrentReadoutCache
from opencood.models.sub_modules.int_packed_current_readout import PackedCurrentReadoutCache,pack_readout_state


class TestPackedCurrentReadout(unittest.TestCase):
    @torch.no_grad()
    def test_independent_sequence_views_ownership_and_cost(self):
        torch.manual_seed(303);source=Readout().eval();repair,_=current_readout_from_mature(source,8,False)
        bounds=[-16,-12,-3,16,12,3]
        plain=LocalCurrentReadoutCache(source,repair,bounds,.5)
        packed=PackedCurrentReadoutCache(source,repair,bounds,.5)
        left=right=None;current=torch.randn(1,16,24,32)
        for i in range(12):
            pose=np.eye(4);angle=.02*(i%4);pose[0,0]=pose[1,1]=np.cos(angle)
            pose[0,1]=-np.sin(angle);pose[1,0]=np.sin(angle);pose[0,3]=.3*(i%3)
            meta=dict(segment='a' if i<9 else 'b',timestamp_us=100000+i*50000,pose=pose)
            old=right;before=None if old is None else old['packed'].clone()
            a,left,ai=plain.step(current,left,meta);b,right,bi=packed.step(current,right,meta)
            self.assertTrue(torch.allclose(a,b,atol=2e-5,rtol=2e-5))
            self.assertTrue(torch.equal(plain.last_mask,packed.last_mask))
            self.assertTrue(torch.equal(left['reference'],right['reference']))
            for key in ('readout_macs_saved','readout_extra_linear_macs','readout_state_bytes'):
                self.assertEqual(ai[key],bi[key])
            self.assertEqual(right['value'].storage().data_ptr(),right['packed'].storage().data_ptr())
            self.assertEqual(right['reference'].storage().data_ptr(),right['packed'].storage().data_ptr())
            confidence=torch.zeros_like(current[:,:1]);confidence[:,:,10,12]=.8
            left['confidence']=confidence;packed.set_confidence(right,confidence)
            self.assertTrue(torch.equal(right['confidence'],confidence))
            if old is not None:self.assertTrue(torch.equal(old['packed'],before))
            current[:,:,8:10,9:11]+=.1
        self.assertEqual(packed.geometry_bytes,24*32*3*4)

    @torch.no_grad()
    def test_same_state_adapter_and_confidence_requirement(self):
        source=Readout().eval();repair,_=current_readout_from_mature(source,8,False)
        bounds=[-16,-12,-3,16,12,3];plain=LocalCurrentReadoutCache(source,repair,bounds,.5)
        cache=PackedCurrentReadoutCache(source,repair,bounds,.5)
        x=torch.randn(1,16,24,32);meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        _,state,_=plain.step(x,None,meta);state['confidence']=torch.zeros_like(x[:,:1])
        adapted=pack_readout_state(state);snapshot=state['value'].clone()
        a,_,_=plain.step(x,state,dict(meta,timestamp_us=50001));b,new,_=cache.step(x,adapted,dict(meta,timestamp_us=50001))
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(state['value'],snapshot))
        with self.assertRaises(ValueError):cache.step(x,new,dict(meta,timestamp_us=100001))

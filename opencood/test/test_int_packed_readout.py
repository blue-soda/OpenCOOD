import unittest
import numpy as np
import torch
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.models.sub_modules.int_packed_readout import PackedReadoutCache


class TestPackedReadout(unittest.TestCase):
    @torch.no_grad()
    def test_same_policy_moving_edges_resets_and_owned_storage(self):
        torch.manual_seed(303);model=Readout().eval();bounds=[-6,-5,-3,6,5,3]
        original=LocalReadoutCache(model,bounds,.25,row_chunk_size=32768)
        packed=PackedReadoutCache(model,bounds,.25,row_chunk_size=32768)
        state=None;reference=None;x=torch.randn(1,16,10,12)
        for index in range(9):
            pose=np.eye(4);pose[0,3]=index*.5
            meta=dict(segment='b' if index>=7 else 'a',timestamp_us=1+50000*index+(300000 if index>=6 else 0),pose=pose)
            current=x.clone();current[:,:,2:5,4:7]+=index*.3
            saved=state['packed'].clone() if state else None
            output,new,info=packed.step(current,state,meta)
            expected,other,control=original.step(current,reference,meta)
            self.assertTrue(torch.equal(packed.last_mask,original.last_mask))
            self.assertTrue(torch.allclose(output,expected,atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.equal(new['reference'],other['reference']))
            self.assertEqual(info['readout_extra_linear_macs'],control['readout_extra_linear_macs'])
            self.assertEqual(info['readout_refresh_reason'],control['readout_refresh_reason'])
            self.assertEqual(new['value'].storage().data_ptr(),new['packed'].storage().data_ptr())
            self.assertEqual(new['reference'].storage().data_ptr(),new['packed'].storage().data_ptr())
            self.assertEqual(new['confidence'].storage().data_ptr(),new['packed'].storage().data_ptr())
            confidence=torch.zeros_like(current[:,:1]);confidence[:,:,6,5]=.8
            packed.set_confidence(new,confidence);other['confidence']=confidence
            if state:self.assertTrue(torch.equal(state['packed'],saved))
            self.assertEqual(new['packed'].numel()*4,info['readout_state_bytes']+confidence.numel()*4)
            # Use identical previous values to isolate execution/representation.
            reference=dict(new);state=new
        self.assertGreater(packed.geometry_bytes,0)

    @torch.no_grad()
    def test_requires_confidence_and_matches_masked_dense(self):
        model=Readout().eval();bounds=[-6,-5,-3,6,5,3]
        cache=PackedReadoutCache(model,bounds,.25);x=torch.randn(1,16,10,12)
        meta=dict(segment='a',timestamp_us=1,pose=np.eye(4));_,state,_=cache.step(x,None,meta)
        with self.assertRaises(ValueError):cache.step(x,state,dict(meta,timestamp_us=50001))
        cache.set_confidence(state,torch.zeros_like(x[:,:1]));saved=state['packed'].clone()
        for current in (x,x+10):
            selected,_,info=cache.step(current,state,dict(meta,timestamp_us=50001))
            cache.execution='masked_dense'
            dense,_,_=cache.step(current,state,dict(meta,timestamp_us=50001));cache.execution='selected'
            self.assertTrue(torch.allclose(selected,dense,atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.equal(state['packed'],saved))
            if torch.equal(current,x):self.assertEqual(info['readout_extra_linear_macs'],0)

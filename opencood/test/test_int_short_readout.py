import unittest
import numpy as np
import torch
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.models.sub_modules.int_short_readout import ShortReadoutState


class TestShortReadout(unittest.TestCase):
    def test_recursive_state_and_discarded_teacher(self):
        torch.manual_seed(303);readout=Readout().eval()
        for p in readout.parameters():p.requires_grad_(False)
        repair=LightRepair(24,4,current_channels=16).eval()
        with torch.no_grad():repair.net[-1].bias.fill_(1.)
        cache=ShortReadoutState(readout,repair,[-6,-5,-3,6,5,3]);x=torch.randn(1,16,10,12)
        meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        cache.training_targets=True;start,state,info=cache.step(x,None,meta);initial=state['value'].clone()
        for index in (1,2):
            m=dict(meta,timestamp_us=1+index*50000);previous=state
            value,state,info=cache.step(x,previous,m)
            self.assertTrue(torch.allclose(value,initial+index,atol=1e-6))
            cache.training_targets=False
            with torch.no_grad():other,_,inference=cache.step(x,previous,m)
            cache.training_targets=True
            self.assertTrue(torch.equal(value,other));self.assertNotIn('readout_target',inference)
            self.assertTrue(torch.equal(info['readout_target'],initial))
        value.sum().backward();self.assertTrue(torch.all(repair.net[-1].bias.grad>0))
        self.assertTrue(all(p.grad is None for p in readout.parameters()))
        value,_,info=cache.step(x,state,dict(meta,timestamp_us=150001))
        self.assertEqual(info['readout_refresh_reason'],'period');self.assertTrue(torch.equal(value,initial))
        with self.assertRaises(ValueError):cache.step(x,state,dict(meta,timestamp_us=100001))
        _,_,info=cache.step(x,state,dict(meta,timestamp_us=400001))
        self.assertEqual(info['readout_refresh_reason'],'age')
        cache.training_targets=False
        with self.assertRaises(ValueError):cache.step(x,None,meta)

    @torch.no_grad()
    def test_period1_exact_and_default_light_repair_compatibility(self):
        readout=Readout().eval();repair=LightRepair(24,4,current_channels=16).eval()
        cache=ShortReadoutState(readout,repair,[-6,-5,-3,6,5,3],1)
        x=torch.randn(1,16,10,12);meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        value,state,_=cache.step(x,None,meta)
        actual,_,info=cache.step(x,state,dict(meta,timestamp_us=50001))
        self.assertTrue(torch.equal(actual,readout.tail(readout.spatial(x))))
        self.assertEqual(info['readout_macs_saved'],0)
        legacy=LightRepair(96,32);packed=torch.randn(1,225,4,4)
        self.assertTrue(torch.equal(legacy(packed),packed[:,128:224]))

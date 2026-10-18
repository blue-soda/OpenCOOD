import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import (UnifiedBEVCore,LocalStateUpdate,
    UnifiedBEVCoreV2,ResidualStateUpdate)
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class UnifiedBEVTests(unittest.TestCase):
    def test_local_update_crop_parity_and_unchanged_regions(self):
        torch.manual_seed(8)
        module=LocalStateUpdate(evidence=8,channels=8).eval()
        x=torch.randn(1,17,19,27);x[:,-1:]=1
        old=x[:,8:16].clone();mask=torch.zeros(1,1,19,27,dtype=torch.bool)
        mask[...,0,0]=True;mask[...,9,13]=True;mask[...,-1,-1]=True
        result,written,stats=refresh_tiles(module,x,old,mask,1,2,4,allow_fallback=False)
        expected=module(x);selected=written.expand_as(result)
        self.assertTrue(torch.allclose(result[selected],expected[selected],atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(result[~selected],old[~selected]))
        self.assertLess(stats['actual_write_fraction'],1.)

    def test_recurrent_gradients_reset_and_causal_order(self):
        torch.manual_seed(7)
        core=UnifiedBEVCore()
        bounds=[-8.,-6.,-3.,8.,6.,3.]
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4),reset_before=False)
        first=torch.randn(1,64,12,16,requires_grad=True)
        _,state,_=core.step(first,None,meta,bounds,False)
        next_meta=dict(meta,timestamp_us=200000)
        pred,next_state,info=core.step(torch.randn_like(first),state,next_meta,bounds,False)
        pred['rm'].square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()),0.)
        self.assertEqual(tuple(next_state['value'].shape),(1,128,6,8))
        self.assertIsNone(info['reset'])
        with self.assertRaises(ValueError):core.step(first,next_state,next_meta,bounds)
        reset_meta=dict(next_meta,segment='b')
        a,_,info=core.step(first,next_state,reset_meta,bounds)
        b,detached,_=core.step(first,None,reset_meta,bounds)
        self.assertEqual(info['reset'],'scene')
        self.assertTrue(torch.equal(a['rm'],b['rm']))
        self.assertIsNone(detached['value'].grad_fn)


class UnifiedBEVV2Tests(unittest.TestCase):
    def test_crop_parity_with_channel_normalization_and_global_edges(self):
        torch.manual_seed(17)
        module=ResidualStateUpdate(evidence=16,channels=16).eval()
        packed=torch.randn(1,33,21,29);packed[:,-1:]=(torch.rand(1,1,21,29)>.15).float()
        old=packed[:,16:32].clone();mask=torch.zeros(1,1,21,29,dtype=torch.bool)
        mask[...,0,0]=True;mask[...,10,14]=True;mask[...,-1,-1]=True
        result,written,stats=refresh_tiles(module,packed,old,mask,1,module.radius,4,allow_fallback=False)
        expected=module(packed);selected=written.expand_as(result)
        self.assertTrue(torch.allclose(result[selected],expected[selected],atol=2e-6,rtol=1e-5))
        self.assertTrue(torch.equal(result[~selected],old[~selected]))
        self.assertLess(stats['actual_write_fraction'],1.)

    def test_newly_visible_cells_ignore_invalid_history(self):
        module=ResidualStateUpdate(evidence=16,channels=16).eval()
        # Invalid history must be zero-filled by the warp before forming packed input.
        packed=torch.randn(1,33,7,9);packed[:,16:32]=0;packed[:,-1:]=0
        residual,_=module.net(packed).chunk(2,dim=1)
        self.assertTrue(torch.equal(module(packed),module.current(packed[:,:16])+residual))

    def test_v2_temporal_gradient_and_reset_control_same_parameters(self):
        torch.manual_seed(21);core=UnifiedBEVCoreV2()
        torch.manual_seed(21);control=UnifiedBEVCoreV2(memory_policy='reset')
        self.assertEqual(sum(p.numel() for p in core.parameters()),sum(p.numel() for p in control.parameters()))
        self.assertTrue(all(torch.equal(v,control.state_dict()[k]) for k,v in core.state_dict().items()))
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4),reset_before=False)
        bounds=[-8.,-6.,-3.,8.,6.,3.]
        first=torch.randn(1,64,13,17,requires_grad=True)
        _,state,_=core.step(first,None,meta,bounds,False)
        current=torch.randn_like(first);next_meta=dict(meta,timestamp_us=200000)
        prediction,new,_=core.step(current,state,next_meta,bounds,False)
        self.assertEqual(tuple(new['value'].shape),(1,256,7,9))
        prediction['rm'].square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):core.step(current,new,next_meta,bounds)
        with torch.no_grad():
            pred,_,_=control.step(current,state,next_meta,bounds)
            isolated,_,_=control.step(current,None,next_meta,bounds)
            self.assertTrue(torch.equal(pred['rm'],isolated['rm']))
            reset_meta=dict(next_meta,reset_before=True)
            reset,_,_=core.step(current,state,reset_meta,bounds)
            isolated,_,_=core.step(current,None,reset_meta,bounds)
            self.assertTrue(torch.equal(reset['rm'],isolated['rm']))


if __name__=='__main__':unittest.main()

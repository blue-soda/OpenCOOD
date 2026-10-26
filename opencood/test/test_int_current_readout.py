import unittest
import numpy as np
import torch
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_short_readout import ShortReadoutState


class TestCurrentReadout(unittest.TestCase):
    @torch.no_grad()
    def test_full_rank_reproduces_mature_readout_and_frozen_bn(self):
        torch.manual_seed(303);source=Readout().eval()
        for bn in (source.tail[0],source.tail[3]):
            bn.running_mean.normal_();bn.running_var.uniform_(.5,2.)
            bn.weight.uniform_(.5,1.5);bn.bias.normal_()
        repair,info=current_readout_from_mature(source,24,False)
        current=torch.randn(1,16,10,12);packed=torch.cat((current,torch.randn(1,24,10,12),torch.ones(1,1,10,12)),1)
        self.assertTrue(torch.allclose(repair(packed),source.tail(source.spatial(current)),atol=2e-5,rtol=2e-5))
        self.assertAlmostEqual(info['second_svd_energy'],1.,places=6)

    def test_same_fresh_initialization_gating_and_uncovered_cells(self):
        source=Readout().eval();plain,_=current_readout_from_mature(source,8,False)
        gated,_=current_readout_from_mature(source,8,True)
        for left,right in zip(plain.net.parameters(),gated.net.parameters()):self.assertTrue(torch.equal(left,right))
        current=torch.randn(1,16,10,12);old=torch.randn(1,24,10,12);valid=torch.zeros(1,1,10,12)
        packed=torch.cat((current,old,valid),1);fresh=plain(packed)
        self.assertTrue(torch.equal(gated(packed),fresh))
        packed[:,-1:]=1.;value=gated(packed)
        self.assertTrue(torch.allclose(value,.475*old+.525*fresh,atol=1e-6))
        value.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gated.parameters()))
        self.assertGreater(float(gated.gate.weight.grad.abs().sum()),0.)

    @torch.no_grad()
    def test_cache_complete_writeback_and_actual_conv_mac_accounting(self):
        source=Readout().eval();repair,_=current_readout_from_mature(source,8,True)
        cache=ShortReadoutState(source,repair,[-6,-5,-3,6,5,3]);current=torch.randn(1,16,10,12)
        meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        _,state,_=cache.step(current,None,meta);calls=[];handles=[]
        for module in repair.modules():
            if isinstance(module,torch.nn.Conv2d):
                handles.append(module.register_forward_hook(lambda m,i,o:calls.append(o.shape[-2]*o.shape[-1]*m.weight.numel())))
        try:value,new,info=cache.step(current,state,dict(meta,timestamp_us=50001))
        finally:
            for h in handles:h.remove()
        full=current.shape[-2]*current.shape[-1]*sum(m.weight.numel() for m in (source.spatial,source.tail[2][0],source.tail[2][1]))
        self.assertEqual(sum(calls),full-info['readout_macs_saved'])
        self.assertTrue(torch.equal(new['value'],value))
        cache.training_targets=True
        other,other_state,_=cache.step(current,state,dict(meta,timestamp_us=50001))
        self.assertTrue(torch.equal(value,other));self.assertTrue(torch.equal(new['value'],other_state['value']))


if __name__=='__main__':unittest.main()

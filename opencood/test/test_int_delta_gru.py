import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory, make_fusion_config
from opencood.models.sub_modules.int_delta_gru import INTDeltaGRUFusion, INTDeltaGRUMemory


def meta(t, x=0., scene='a'):
    p=np.eye(4); p[0,3]=x
    return dict(timestamp_us=t,pose=p.tolist(),segment=scene)


class Constant(nn.Module):
    def __init__(self,v): super().__init__(); self.v=v
    def forward(self,x): return torch.full_like(x[:,:1],self.v)


class DeltaGRUTests(unittest.TestCase):
    def fusion(self):
        f=INTDeltaGRUFusion(make_fusion_config('gru',1)).eval()
        f.reset_gate_conv=Constant(.5); f.update_gate_conv=Constant(0.)
        f.candidate_conv=Constant(.1)
        f.post_fusion_conv=nn.Conv2d(3,1,3,padding=1,bias=False)
        with torch.no_grad():
            f.post_fusion_conv.weight.zero_(); f.post_fusion_conv.weight[0,0,1,1]=1
        return f

    def test_small_residual_accumulates_both_polarities(self):
        for sign in (-1,1):
            f=self.fusion(); f.candidate_conv=Constant(sign*.1)
            state=torch.zeros(1,6,4,8);state[:,-1]=1
            x=torch.zeros(1,1,4,8)
            for i in range(3):
                out,state=f(x,[state],.05,False)
                if i<2:self.assertEqual(float(out.abs().sum()),0.)
            self.assertTrue(torch.allclose(out,torch.full_like(out,sign*.25)))
            self.assertTrue(torch.allclose(state[:,:1],torch.full_like(out,sign*.05),atol=1e-6))

    def test_zero_innovation_reuses_readout_without_erasing_empty_cells(self):
        f=self.fusion();x=torch.zeros(1,1,4,8)
        state=torch.zeros(1,6,4,8);state[:,1]=.1;state[:,2]=.7;state[:,-1]=1
        previous=state.clone();out,new=f(x,[state],.05)
        self.assertTrue(torch.equal(out,state[:,2:3]));self.assertTrue(torch.equal(state,previous))
        self.assertEqual(f.last_diagnostics['readout_refresh_fraction'],0.)
        self.assertTrue(torch.allclose(new[:,1:2],state[:,1:2]))

    def test_expiry_bootstrap_and_halo(self):
        f=self.fusion();state=torch.zeros(1,6,8,8);state[:,-1]=1
        state[:,3]=.99
        out,new=f(torch.zeros(1,1,8,8),[state],.02)
        self.assertEqual(f.last_diagnostics['forced_refresh_fraction'],1.)
        self.assertTrue(torch.allclose(out,torch.full_like(out,.1)))
        self.assertEqual(float(new[:,3].sum()),0.)
        state.zero_();state[:,-1]=1;state[0,0,4,4]=.2
        f(torch.zeros(1,1,8,8),[state],.05)
        self.assertEqual(f.last_diagnostics['readout_refresh_fraction'],9/64)

    def test_budget_parameters_and_temporal_gradient(self):
        region=[0,0,-1,8,4,1]
        torch.manual_seed(303);a=INTFeatureMemory('gru',4,region).eval()
        torch.manual_seed(303);m=INTDeltaGRUMemory(4,region).eval()
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
        for detach in (False,True):
            x=torch.randn(1,4,4,8,requires_grad=True)
            _,s,info=m.step(x,None,meta(1000000),detach_state=detach)
            self.assertEqual(info['state_bytes'],15*4*8*4)
            y,_,_=m.step(torch.randn_like(x),s,meta(1100000),detach_state=False)
            y.square().sum().backward()
            if detach:self.assertIsNone(x.grad)
            else:self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_warp_boundary_and_scene_reset(self):
        m=INTDeltaGRUMemory(1,[0,0,-1,8,4,1]).eval();m.fusion=self.fusion()
        x=torch.zeros(1,1,4,8)
        _,s,_=m.step(x,None,meta(1000000));before=s['value'].clone()
        _,new,info=m.step(x,s,meta(1100000,x=1.))
        self.assertEqual(info['forced_refresh_fraction'],1/8)
        self.assertTrue(torch.equal(before,s['value']))
        for t,scene in ((1100000,'b'),(1400000,'a')):
            y,_,_=m.step(x,s,meta(t,scene=scene));fresh,_,_=m.step(x,None,meta(t,scene=scene))
            self.assertTrue(torch.equal(y,fresh))


if __name__=='__main__':unittest.main()

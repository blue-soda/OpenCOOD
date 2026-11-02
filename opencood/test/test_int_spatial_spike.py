import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_spatial_spike import SpatialActivationMemory,SpatialSpikeAnchor,deliveries3x3
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector


def meta(t,scene='s'):return dict(timestamp_us=t,segment=scene,pose=np.eye(4).tolist())


class SpatialSpikeTests(unittest.TestCase):
    def memory(self,mode):return SpatialActivationMemory(mode,[1.],[0,0,-1,4,2,1])

    def test_exact_fanout_including_degenerate_shapes(self):
        for h,w in [(1,1),(1,4),(2,4),(5,7)]:
            active=torch.arange(h*w).reshape(1,1,h,w)%2==0
            brute=sum(sum(0<=y+dy<h and 0<=x+dx<w for dy in (-1,0,1) for dx in (-1,0,1)) for y in range(h) for x in range(w) if active[0,0,y,x])*3
            self.assertEqual(deliveries3x3(active,3),brute)

    def test_dynamics_reset_and_no_mutation(self):
        for mode in ('lif','leaky'):
            m=self.memory(mode);a=torch.full((1,1,2,4),.8)
            first,state,_=m.step(a,None,meta(1000000));old=state['value'].clone()
            later,new,info=m.step(a,state,meta(1100000));reset,_,_=m.step(a,state,meta(1100000),reset=True)
            self.assertTrue(torch.equal(old,state['value']));self.assertTrue(torch.equal(first,reset));self.assertGreater(float(later.sum()),float(first.sum()))
            self.assertEqual(info['state_bytes'],32);self.assertTrue(info['warp']);self.assertIsNone(new['value'].grad_fn)

    def test_gap_scene_causality_and_invalid_scale(self):
        m=self.memory('lif');a=torch.ones(1,1,2,4);_,s,_=m.step(a,None,meta(1000000))
        self.assertEqual(m.step(a,s,meta(1400000))[2]['reset'],'gap')
        self.assertEqual(m.step(a,s,meta(10,'other'))[2]['reset'],'scene')
        with self.assertRaises(ValueError):m.step(a,s,meta(1000000),reset=True)
        with self.assertRaises(ValueError):SpatialActivationMemory('lif',[0.],[0,0,-1,4,2,1])

    def test_translation_warps_membrane(self):
        m=self.memory('leaky');a=torch.zeros(1,1,2,4);a[:,:,:,1]=1
        _,s,_=m.step(a,None,meta(1000000));later=meta(1100000);later['pose'][0][3]=1
        out,_,_=m.step(torch.zeros_like(a),s,later)
        self.assertTrue(torch.all(out[:,:,:,0]>0));self.assertEqual(float(out[:,:,:,1:].abs().sum()),0.)

    def test_input_credit_is_cut_at_membrane_boundary(self):
        for mode in ('lif','leaky'):
            for cut in (False,True):
                m=self.memory(mode);a=torch.full((1,1,2,4),.8,requires_grad=True)
                _,s,_=m.step(a,None,meta(1000000),detach_state=cut)
                out,_,_=m.step(a.detach().requires_grad_(True),s,meta(1100000),detach_state=False);out.sum().backward()
                if cut:self.assertIsNone(a.grad)
                else:self.assertGreater(float(a.grad.abs().sum()),0.)

    def test_ann_wrapper_is_exact_and_only_target_weight_open(self):
        from test_int_pyramid_state import make_pair
        torch.manual_seed(303);_,base=make_pair();strict=AnchoredLatePyramidDetector(base);wrapped=SpatialSpikeAnchor(base,'ann',torch.ones(256))
        self.assertEqual(sum(p.numel() for p in wrapped.parameters()),sum(p.numel() for p in strict.parameters()))
        self.assertEqual([n for n,p in wrapped.named_parameters() if p.requires_grad],['core.coarse_encoder.3.conv2.weight'])
        a=b=None;x=torch.rand(1,64,24,32)
        with torch.no_grad():
            for t in (1000000,1100000):
                expected,a,_=strict.step_spatial(x,a,meta(t));actual,b,info=wrapped.step_spatial(x,b,meta(t))
                self.assertTrue(all(torch.equal(v,actual[k]) for k,v in expected.items()))
                self.assertTrue(torch.equal(a['coarse'],b['coarse']));self.assertEqual(info['cache_bytes'],sum(a[k].numel()*4 for k in ('value','coarse'))+a['anchor']['value'].numel()*4)
                self.assertIsNone(b['spatial_membrane'])


if __name__=='__main__':unittest.main()

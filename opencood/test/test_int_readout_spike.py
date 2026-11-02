import unittest
from unittest import mock
import torch
from opencood.models.sub_modules.int_spatial_spike import SpatialActivationMemory
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_temporal_feature import detach_temporal_state
from test_int_spatial_spike import meta
from test_int_pyramid_state import make_pair


class ReadoutSpikeTests(unittest.TestCase):
    def memory(self,mode='leaky'):
        return SpatialActivationMemory(mode,[1.],[0,0,-1,4,2,1],storage_dtype=torch.float16)

    def test_storage_quantizes_only_state_and_warp_uses_float32(self):
        m=self.memory();x=torch.full((1,1,2,4),.8001)
        y,s,info=m.step(x,None,meta(1000000));self.assertTrue(torch.equal(x,y))
        self.assertEqual(s['value'].dtype,torch.float16);self.assertEqual(info['state_bytes'],16)
        self.assertGreater(info['storage_quantization_abs_max'],0)
        from opencood.models.sub_modules.int_feature_memory import warp_bev
        with mock.patch('opencood.models.sub_modules.int_spatial_spike.warp_bev',wraps=warp_bev) as warp:
            m.step(x,s,meta(1100000));self.assertEqual(warp.call_args[0][0].dtype,torch.float32)
        self.assertEqual(info['decode_fp32_tensor_bytes'],0)

    def test_overflow_rejected_without_clipping_and_invalid_dtype(self):
        with self.assertRaises(RuntimeError):self.memory().step(torch.full((1,1,2,4),70000.),None,meta(1))
        with self.assertRaises(ValueError):SpatialActivationMemory('lif',[1.],[0,0,-1,4,2,1],storage_dtype=torch.bfloat16)
        with self.assertRaises(ValueError):self.memory().step(torch.ones(1,1,2,4).half(),None,meta(1))

    def test_half_state_keeps_gradient_and_cut_resets(self):
        for mode in ('lif','leaky'):
            for cut in ('none','detach','reset','scene','gap'):
                m=self.memory(mode);x=torch.full((1,1,2,4),.8,requires_grad=True)
                _,s,_=m.step(x,None,meta(1000000),detach_state=False);old=s['value'].detach().clone()
                if cut=='detach':s=detach_temporal_state(s)
                target=meta(1400000 if cut=='gap' else 1100000,'other' if cut=='scene' else 's')
                y,_,_=m.step(torch.ones_like(x).requires_grad_(),s,target,reset=cut=='reset',detach_state=False)
                y.sum().backward();self.assertTrue(torch.equal(s['value'],old))
                if cut=='none':self.assertGreater(float(x.grad.abs().sum()),0)
                else:self.assertTrue(x.grad is None or float(x.grad.abs().sum())==0)

    def test_half_state_translation_and_causality(self):
        m=self.memory();x=torch.zeros(1,1,2,4);x[:,:,:,1]=1
        _,s,_=m.step(x,None,meta(1000000));moved=meta(1100000);moved['pose'][0][3]=1
        out,_,_=m.step(torch.zeros_like(x),s,moved)
        self.assertGreater(float(out[:,:,:,0].sum()),0);self.assertEqual(float(out[:,:,:,1:].sum()),0)
        with self.assertRaises(ValueError):m.step(x,s,meta(1000000),reset=True)

    def test_ann_exact_cache_and_learning_scope(self):
        torch.manual_seed(303);_,base=make_pair();strict=AnchoredLatePyramidDetector(base)
        m=ReadoutSpikeAnchor(base,'ann',torch.ones(256),torch.float16);a=b=None
        self.assertEqual(set(n for n,p in m.named_parameters() if p.requires_grad),set(m.trainable_names))
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),1475072)
        self.assertTrue(all(not x.training for x in m.modules() if isinstance(x,torch.nn.BatchNorm2d)))
        with torch.no_grad():
            for i,t in enumerate((1000000,1100000,1500000)):
                x=torch.rand(1,64,24,32);md=meta(t);md['reset_before']=i==2
                p,a,ia=strict.step_spatial(x,a,md);q,b,ib=m.step_spatial(x,b,md)
                self.assertTrue(all(torch.equal(v,q[k]) for k,v in p.items()))
                for k in ('value','coarse'):self.assertTrue(torch.equal(a[k],b[k]))
                self.assertTrue(torch.equal(a['anchor']['value'],b['anchor']['value']))
                self.assertIsNone(b['readout_membrane']);self.assertEqual(ia['cache_bytes'],ib['cache_bytes'])
                self.assertFalse(any(x._forward_pre_hooks for x in m.modules()))

    def test_membrane_does_not_feed_back_into_ann_states(self):
        torch.manual_seed(303);_,base=make_pair();strict=AnchoredLatePyramidDetector(base)
        m=ReadoutSpikeAnchor(base,'lif',torch.ones(256),torch.float16);a=b=None
        with torch.no_grad():
            for t in (1000000,1100000):
                x=torch.rand(1,64,24,32);_,a,ia=strict.step_spatial(x,a,meta(t));_,b,ib=m.step_spatial(x,b,meta(t))
                for k in ('value','coarse'):self.assertTrue(torch.equal(a[k],b[k]))
                self.assertTrue(torch.equal(a['anchor']['value'],b['anchor']['value']))
                self.assertEqual(ib['cache_bytes']-ia['cache_bytes'],256*12*16*2)

    def test_readout_early_membrane_and_both_convolutions_learn(self):
        torch.manual_seed(303);_,base=make_pair()
        for mode in ('lif','leaky'):
            m=ReadoutSpikeAnchor(base,mode,torch.full((256,),.01),torch.float16);s=None
            for i in range(4):
                p,s,_=m.step_spatial(torch.rand(1,64,24,32),s,meta(1000000+i*100000),detach_state=False)
                if i==0:early=s['readout_membrane']['value'];early.retain_grad()
            sum(v.square().sum() for v in p.values()).backward()
            self.assertGreater(float(early.grad.abs().sum()),0)
            for n,param in m.named_parameters():
                if n in m.trainable_names:
                    self.assertTrue(bool(torch.isfinite(param.grad).all()));self.assertGreater(float(param.grad.abs().sum()),0,n)
                else:self.assertIsNone(param.grad)

    def test_exception_removes_hooks(self):
        _,base=make_pair();m=ReadoutSpikeAnchor(base,'lif',torch.ones(256))
        with mock.patch.object(m.activation_memory,'step',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):m.step_spatial(torch.rand(1,64,24,32),None,meta(1))
        self.assertFalse(any(x._forward_pre_hooks for x in m.modules()))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_temporal_feature import residual_feature_memory
from opencood.models.sub_modules.int_temporal_feature import TemporalFeatureAnchor,DifferentiableTemporalFeatureAnchor,detach_temporal_state


def meta(t):
    return dict(timestamp_us=t,segment='s',pose=np.eye(4).tolist(),reset_before=False)


class TemporalFeatureTests(unittest.TestCase):
    def make(self,mode):return residual_feature_memory(mode,1,[0,0,-1,4,2,1],1)

    def test_identity_initialization_and_matched_parameters(self):
        lif,ann=self.make('lif'),self.make('leaky')
        self.assertEqual(sum(p.numel() for p in lif.parameters()),sum(p.numel() for p in ann.parameters()))
        self.assertTrue(all(torch.equal(v,ann.state_dict()[k]) for k,v in lif.state_dict().items()))
        for model in (lif,ann):
            state=None;x=torch.full((1,1,2,4),2.8)
            for t in (1000000,1100000,1200000):
                out,state,info=model.step(x,state,meta(t))
                self.assertTrue(torch.equal(out,torch.zeros_like(x)))
                self.assertEqual(info['state_bytes'],2*x.numel()*4)

    def test_zero_projection_receives_gradient_from_real_history(self):
        model=self.make('lif');x=torch.full((1,1,2,4),2.8)
        _,state,_=model.step(x,None,meta(1000000),detach_state=False)
        out,_,_=model.step(x,state,meta(1100000),detach_state=False)
        out.sum().backward()
        self.assertGreater(float(model.readout.weight.grad.abs().sum()),0)

    def test_history_vs_reset_after_opening_projection(self):
        model=self.make('lif')
        with torch.no_grad():model.readout.weight.fill_(1.)
        x=torch.full((1,1,2,4),2.8)
        _,state,_=model.step(x,None,meta(1000000))
        history,_,_=model.step(x,state,meta(1100000))
        reset,_,_=model.step(x,state,meta(1100000),reset=True)
        self.assertGreater(float(history.sum()),float(reset.sum()))

    def test_differentiable_wrapper_preserves_inference_and_freezes_backbone(self):
        from test_int_pyramid_state import make_pair
        torch.manual_seed(303);_,base=make_pair()
        frozen=TemporalFeatureAnchor(base,'leaky');model=DifferentiableTemporalFeatureAnchor(base,'leaky')
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),82016)
        x=torch.randn(1,64,24,32);a=b=None
        for t in (1000000,1100000):
            with torch.no_grad():expected,a,_=frozen.step_spatial(x,a,meta(t))
            actual,b,_=model.step_spatial(x,b,meta(t),detach_state=False)
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in expected.items()))
        self.assertIsNotNone(b['feature_memory']['value'].grad_fn)

    def test_nested_detach_keeps_values_and_cuts_every_state(self):
        v=torch.ones(2,requires_grad=True)*2
        state=dict(value=v,coarse=v+1,anchor=dict(value=v+2,skipped=1),feature_memory=dict(value=v+3),pose=np.eye(4),extra=(v,))
        detached=detach_temporal_state(state)
        for a,b in [(state['value'],detached['value']),(state['coarse'],detached['coarse']),
                    (state['anchor']['value'],detached['anchor']['value']),(state['feature_memory']['value'],detached['feature_memory']['value'])]:
            self.assertTrue(torch.equal(a,b));self.assertIsNone(b.grad_fn)
        self.assertIs(detached['pose'],state['pose']);self.assertFalse(detached['extra'][0].requires_grad)

    def test_detector_future_credit_cut_by_detach_and_scene_reset(self):
        from test_int_pyramid_state import make_pair
        torch.manual_seed(303);_,base=make_pair();model=DifferentiableTemporalFeatureAnchor(base,'leaky')
        with torch.no_grad():model.core.temporal_feature.readout.weight.fill_(.001)
        for branch in ('connected','detach','scene'):
            model.zero_grad(set_to_none=True)
            x=torch.rand(1,64,24,32,requires_grad=True)
            _,state,_=model.step_spatial(x,None,meta(1000000),detach_state=False)
            previous=state['feature_memory']['value'];previous.retain_grad()
            if branch=='detach':state=detach_temporal_state(state)
            later=meta(1100000)
            if branch=='scene':later['segment']='new'
            pred,_,_=model.step_spatial(x.detach(),state,later,detach_state=False)
            (pred['psm'].square().mean()+pred['rm'].square().mean()).backward()
            if branch=='connected':
                self.assertGreater(float(x.grad.abs().sum()),0)
                self.assertGreater(float(previous.grad.abs().sum()),0)
            else:self.assertIsNone(x.grad);self.assertIsNone(previous.grad)
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('core.temporal_feature.')))

    def test_reused_anchor_graph_is_retained_then_detached(self):
        from test_int_pyramid_state import make_pair
        torch.manual_seed(303);_,base=make_pair();model=DifferentiableTemporalFeatureAnchor(base,'leaky')
        x=torch.rand(1,64,24,32)
        _,first,_=model.step_spatial(x,None,meta(1000000),detach_state=False)
        _,second,info=model.step_spatial(x,first,meta(1100000),detach_state=False)
        self.assertIsNone(info['late_refresh'])
        self.assertIs(second['anchor']['value'],first['anchor']['value'])
        self.assertIsNotNone(second['anchor']['value'].grad_fn)
        _,third,_=model.step_spatial(x,second,meta(1200000),detach_state=True)
        self.assertTrue(all(not v.requires_grad for v in (third['value'],third['coarse'],third['anchor']['value'],third['feature_memory']['value'])))


if __name__=='__main__':unittest.main()

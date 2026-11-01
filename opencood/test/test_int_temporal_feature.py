import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_temporal_feature import residual_feature_memory


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


if __name__=='__main__':unittest.main()

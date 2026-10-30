import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_two_scale import TwoScaleDetector


def block(ci, co, stride):
    return BasicBlock(ci,co,stride,nn.Sequential(nn.Conv2d(ci,co,1,stride,bias=False),nn.BatchNorm2d(co)))


class TwoScaleTests(unittest.TestCase):
    def model(self, policy='recurrent'):
        teacher=SimpleNamespace(pillar_vfe=nn.Identity(),scatter=nn.Identity(),use_dir=False,
            cls_head=nn.Conv2d(256,2,1),reg_head=nn.Conv2d(256,14,1),
            backbone=SimpleNamespace(resnet=SimpleNamespace(layer0=nn.Sequential(block(64,128,2)),
                layer1=nn.Sequential(block(128,256,2),BasicBlock(256,256)))))
        return TwoScaleDetector(teacher,{'model':{'args':{'lidar_range':[-4,-4,-1,4,4,1]}}},policy)

    def test_initial_policy_equivalence_and_state_ownership(self):
        torch.manual_seed(303)
        model=self.model();reset=copy.deepcopy(model);reset.memory_policy='reset'
        x=torch.randn(1,64,16,20);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta)
        before={k:v.clone() for k,v in state.items() if torch.is_tensor(v)}
        meta=dict(meta,timestamp_us=200000)
        a,_,_=model.step_spatial(x,state,meta);b,_,_=reset.step_spatial(x,state,meta)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        self.assertTrue(all(torch.equal(state[k],v) for k,v in before.items()))
        self.assertEqual(state['value'].shape,(1,128,8,10))
        self.assertEqual(state['coarse'].shape,(1,256,4,5))
        with self.assertRaises(ValueError):model.step_spatial(x,state,dict(meta,timestamp_us=100000))
        _,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=500000))
        self.assertEqual(info['reset'],'gap')

    def test_temporal_gradients_and_detach(self):
        model=self.model().train()
        nn.init.normal_(model.core.fine_update.history.weight,std=.01)
        nn.init.normal_(model.core.coarse_update.history.weight,std=.01)
        x=torch.randn(1,64,16,20);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta,False)
        for k in ('value','coarse'):state[k].retain_grad()
        pred,_,_=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
        pred['psm'].square().mean().backward()
        for k in ('value','coarse'):self.assertGreater(float(state[k].grad.abs().sum()),0)
        _,detached,_=model.step_spatial(x,None,meta,True)
        self.assertTrue(all(detached[k].grad_fn is None for k in ('value','coarse')))


if __name__=='__main__':unittest.main()

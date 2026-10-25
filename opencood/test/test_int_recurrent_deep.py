import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepState


class AddUpdate(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,x):return x[:,128:129]+self.scale


class TestRecurrentDeep(unittest.TestCase):
    def test_recursive_writeback_refresh_and_gradient(self):
        repair=AddUpdate();cache=RecurrentDeepState(repair,[-2,-2,-1,2,2,1],6)
        h=torch.zeros(1,128,4,4);calls=[]
        def target():calls.append(1);return torch.zeros(1,1,4,4)
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4));state=None
        for index in range(7):
            value,state,info=cache.step(h,state,dict(meta,timestamp_us=100000+50000*index),target,detach=False)
            self.assertEqual(float(value.mean()),index%6)
            if index==3:
                grad=torch.autograd.grad(value.sum(),repair.scale,retain_graph=True)[0]
                self.assertEqual(float(grad),48.)
        self.assertEqual(len(calls),2)
        self.assertEqual(info['deep_refresh_reason'],'period')
        with self.assertRaises(ValueError):cache.step(h,state,dict(meta,timestamp_us=400000),target)
        _,_,info=cache.step(h,state,dict(meta,timestamp_us=1000000),target)
        self.assertEqual(info['deep_refresh_reason'],'age')
        _,_,info=cache.step(h,state,dict(meta,segment='b'),target)
        self.assertEqual(info['deep_refresh_reason'],'reset')

    def test_detach_and_no_mutation(self):
        cache=RecurrentDeepState(AddUpdate(),[-2,-2,-1,2,2,1],3)
        h=torch.zeros(1,128,4,4);meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        _,state,_=cache.step(h,None,meta,lambda:torch.zeros(1,1,4,4))
        value,new,_=cache.step(h,state,dict(meta,timestamp_us=50001),lambda:None)
        self.assertFalse(new['value'].requires_grad)
        self.assertTrue(value.requires_grad)
        self.assertEqual(float(state['value'].sum()),0.)
        self.assertEqual(float(new['value'].sum()),16.)

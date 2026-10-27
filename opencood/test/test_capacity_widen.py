import copy
import unittest
import torch
from torch import nn
from opencood.utils.e3dsnn_capacity import widen_backend


class StrictConfig(dict):
    __getattr__=dict.__getitem__


class DummyBEV(nn.Module):
    def __init__(self):
        super().__init__();self.num_bev_features=6
        self.model_cfg={'NUM_FILTERS':[2,4],'NUM_UPSAMPLE_FILTERS':[3,3]}
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(5,2,3,padding=1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),nn.Conv2d(2,2,3,padding=1,bias=False),nn.BatchNorm2d(2)),
                                   nn.Sequential(nn.Conv2d(2,4,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(4))])
        self.deblocks=nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(2,3,1,bias=False),nn.BatchNorm2d(3)),nn.Sequential(nn.ConvTranspose2d(4,3,2,stride=2,bias=False),nn.BatchNorm2d(3))])
    def forward(self,x):
        ys=[]
        for block,up in zip(self.blocks,self.deblocks):x=block(x);ys.append(up(x))
        return torch.cat(ys,dim=1)


class CapacityTest(unittest.TestCase):
    def test_preserves_function_rng_and_breaks_gradient_symmetry(self):
        torch.manual_seed(17);model=nn.Module();model.backbone_2d=DummyBEV()
        model.cls_head=nn.Conv2d(6,2,1);model.reg_head=nn.Conv2d(6,14,1);model.dir_head=nn.Conv2d(6,4,1)
        model.eval();small=copy.deepcopy(model)
        model.backbone_2d.model_cfg=StrictConfig(model.backbone_2d.model_cfg)
        rng=torch.get_rng_state().clone();widen_backend(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        model.eval();x=torch.randn(2,5,10,12)
        a=small.backbone_2d(x);b=model.backbone_2d(x)
        for name in ('cls_head','reg_head','dir_head'):
            self.assertTrue(torch.allclose(getattr(small,name)(a),getattr(model,name)(b),atol=2e-6,rtol=2e-6))
        model.reg_head(b).square().mean().backward()
        grad=model.backbone_2d.blocks[0][0].weight.grad
        self.assertGreater(float((grad[:2]-grad[2:]).abs().max()),0.)


if __name__=='__main__':unittest.main()

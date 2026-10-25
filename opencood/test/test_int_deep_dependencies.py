import unittest
import torch
from torch import nn
from opencood.models.sub_modules.resblock import BasicBlock
from opencood.models.sub_modules.int_deep_dependencies import deep_dependencies


class TestDeepDependencies(unittest.TestCase):
    def test_trace_matches_gradient_and_dense_hook_cost(self):
        def stage(ci,co):
            return nn.Sequential(BasicBlock(ci,co,2,nn.Sequential(nn.Conv2d(ci,co,1,stride=2,bias=False),nn.BatchNorm2d(co))),BasicBlock(co,co))
        second,third=stage(2,3),stage(3,4)
        decode=nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(3,2,2,2,bias=False)),nn.Sequential(nn.ConvTranspose2d(4,2,4,4,bias=False))])
        projection=nn.Conv2d(4,2,1,bias=False)
        root=nn.ModuleList([second,third,decode,projection]).eval()
        with torch.no_grad():
            for m in root.modules():
                if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):m.weight.fill_(.1)
        counts=[];hooks=[]
        def count(m,args,out):
            counts.append(args[0].shape[-2]*args[0].shape[-1]*m.weight.numel() if isinstance(m,nn.ConvTranspose2d)
                          else out.shape[-2]*out.shape[-1]*m.weight.numel())
        for m in root.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):hooks.append(m.register_forward_hook(count))
        x=torch.ones(1,2,32,40,requires_grad=True);y=second(x);z=third(y)
        output=projection(torch.cat((decode[0](y),decode[1](z)),1))
        for h in hooks:h.remove()
        full=torch.ones(1,1,32,40,dtype=torch.bool)
        self.assertEqual(deep_dependencies(second,third,decode,projection,full,(32,40))['macs'],sum(counts))
        self.assertEqual(deep_dependencies(second,third,decode,projection,~full,(32,40))['macs'],0)
        for yy,xx in ((0,0),(16,20),(31,39)):
            mask=torch.zeros_like(full);mask[:,:,yy,xx]=True
            expected=torch.autograd.grad((output*mask).sum(),x,retain_graph=True)[0].abs().sum(1,keepdim=True)>0
            trace=deep_dependencies(second,third,decode,projection,mask,(32,40))
            self.assertTrue(torch.equal(expected,trace['required_h']))
            self.assertLessEqual(trace['macs'],sum(counts))

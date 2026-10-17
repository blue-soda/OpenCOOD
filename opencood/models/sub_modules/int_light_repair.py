"""Short-receptive-field residual repair of independently cached deep features."""
import torch
from torch import nn
import torch.nn.functional as F


class LightRepair(nn.Module):
    def __init__(self,channels,width=64):
        super().__init__();self.channels=channels
        self.net=nn.Sequential(nn.Conv2d(128+channels+1,width,1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1,groups=width),nn.ReLU(),nn.Conv2d(width,channels,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return x[:,128:128+self.channels]+self.net(x)


def repair_input_request(first,old,valid,changed,index):
    stride=2**index
    evidence=F.avg_pool2d(first,stride,stride,ceil_mode=True,count_include_pad=False)
    changed=F.max_pool2d(changed.float(),stride,stride,ceil_mode=True)>0
    request=F.max_pool2d(changed.float(),3,1,1)>0
    assert evidence.shape[-2:]==old.shape[-2:]==request.shape[-2:]
    return torch.cat([evidence,old,valid.to(old.dtype)],1),request | ~valid


def make_repairs():
    return nn.ModuleList([LightRepair(256),LightRepair(512)])

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


def make_repairs(target='stages'):
    if target not in ('stages','decoded'):raise ValueError(target)
    return nn.ModuleList([LightRepair(256)] if target=='decoded' else [LightRepair(256),LightRepair(512)])


def foreground_head_loss(predictions,teacher):
    """Normalize box/direction supervision by teacher foreground mass, not area."""
    confidence=teacher['psm'].detach().sigmoid().square()
    mass=confidence.sum().clamp_min(1.)
    cls=F.smooth_l1_loss(predictions['psm'],teacher['psm'].detach(),reduction='none')
    loss=.1*cls.mean()+.1*(cls*confidence).sum()/mass
    batch,anchors,height,width=confidence.shape
    reg=F.smooth_l1_loss(predictions['rm'],teacher['rm'].detach(),reduction='none',beta=1./9.)
    reg=reg.reshape(batch,anchors,-1,height,width).mean(2)
    loss=loss+(reg*confidence).sum()/mass
    if 'dm' in predictions:
        student=predictions['dm'].reshape(batch,anchors,-1,height,width)
        reference=teacher['dm'].detach().reshape_as(student).softmax(2)
        direction=F.kl_div(student.log_softmax(2),reference,reduction='none').sum(2)
        loss=loss+.1*(direction*confidence).sum()/mass
    return loss

from types import SimpleNamespace
import torch
from torch import nn
from opencood.models.sfr.pillar_fusion import SfrPillarFusion


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(64,8,3,padding=1,stride=2);self.bn=nn.BatchNorm2d(8)
    def forward(self,data):return dict(spatial_features_2d=self.bn(self.conv(data['spatial_features'])))


def test_pillar_decoder_gradient_and_single_bn_fallback():
    sender=SimpleNamespace(bounds=[0,0,-1,8,8,1],backbone=Backbone().eval(),shrink_flag=False,
                           cls_head=nn.Conv2d(8,2,1),reg_head=nn.Conv2d(8,14,1),use_dir=False)
    model=SfrPillarFusion(sender,dict(orientation_conditioned=True)).train()
    ego=torch.randn(1,64,8,8)
    original=sender.cls_head(sender.backbone(dict(spatial_features=ego))['spatial_features_2d'])
    empty,_=model(ego,None);torch.testing.assert_close(empty['psm'],original,atol=0,rtol=0)
    running=model.single_backbone.bn.running_mean.clone()
    features=torch.ones(1,64,requires_grad=True)*5
    output,_=model(ego,(torch.tensor([[3.5,3.5,0.]]),features,torch.tensor([.3]),torch.tensor([.9]),torch.tensor([[0.,1.]])))
    output['psm'].square().sum().backward()
    assert model.adapter.weight.grad.abs().sum()>0 and model.backbone.conv.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.single_backbone.parameters())
    torch.testing.assert_close(running,model.single_backbone.bn.running_mean,atol=0,rtol=0)


def test_diagnostic_components_reconstruct_prediction_without_changing_forward():
    sender=SimpleNamespace(bounds=[0,0,-1,8,8,1],backbone=Backbone().eval(),shrink_flag=False,
                           cls_head=nn.Conv2d(8,2,1),reg_head=nn.Conv2d(8,14,1),use_dir=False)
    model=SfrPillarFusion(sender,dict(orientation_conditioned=False)).eval()
    ego=torch.randn(1,64,8,8)
    received=(torch.tensor([[3.5,3.5,0.]]),torch.ones(1,64),torch.tensor([.3]),torch.tensor([.25]))
    for message in (None,received):
        original,_=model(ego,message)
        output,diagnostic=model(ego,message,return_components=True)
        baseline,decoded,mask=diagnostic['components']
        assert mask.shape[-2:]==output['psm'].shape[-2:]
        for key in output:
            torch.testing.assert_close(output[key],original[key],atol=0,rtol=0)
            torch.testing.assert_close(output[key],baseline[key]+mask*(decoded[key]-baseline[key]),atol=0,rtol=0)

import copy
import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_head_calibration import (
    detection_heads,enable_only_detection_heads,spatial_head_checkpoint,load_spatial_head_checkpoint)


def fixture():
    model=nn.Module();model.use_dir=True
    model.backbone=nn.Module();model.backbone.resnet=nn.Module()
    model.backbone.resnet.layer0=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4))
    for name,channels in [('cls_head',2),('reg_head',14),('dir_head',4)]:
        setattr(model,name,nn.Conv2d(4,channels,1))
    model.eval();return model


class HeadCalibrationTests(unittest.TestCase):
    def test_only_existing_heads_change_and_round_trip_includes_stage(self):
        torch.manual_seed(303);model=fixture();before=copy.deepcopy(model.state_dict())
        count=sum(p.numel() for p in model.parameters());heads=enable_only_detection_heads(model)
        optimizer=torch.optim.AdamW(heads.parameters(),lr=.001)
        x=torch.randn(1,3,5,7);features=model.backbone.resnet.layer0(x)
        self.assertFalse(features.requires_grad)
        loss=sum(head(features).square().mean() for head in heads.values())
        loss.backward();optimizer.step()
        self.assertEqual(count,sum(p.numel() for p in model.parameters()))
        for key,value in model.state_dict().items():
            if key.startswith('backbone.'):self.assertTrue(torch.equal(value,before[key]))
        for head in heads.values():self.assertGreater(float(head.weight.grad.abs().sum()),0.)
        self.assertFalse(torch.equal(model.reg_head.weight,before['reg_head.weight']))
        saved=copy.deepcopy(spatial_head_checkpoint(model,{'source':'fixed'},1));restored=fixture()
        load_spatial_head_checkpoint(restored,saved)
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,restored.state_dict()[key]))
        with torch.no_grad():
            original=heads['reg_head'](model.backbone.resnet.layer0(x))
            actual=restored.reg_head(restored.backbone.resnet.layer0(x))
            self.assertTrue(torch.equal(original,actual))
        with self.assertRaisesRegex(ValueError,'both pointwise'):
            load_spatial_head_checkpoint(restored,{'stage':saved['stage']})

    def test_direction_head_is_optional(self):
        model=fixture();model.use_dir=False;del model.dir_head
        self.assertEqual(set(detection_heads(model)),{'cls_head','reg_head'})


if __name__=='__main__':unittest.main()

"""Calibrate existing pointwise detection heads without changing inference."""
from torch import nn


def detection_heads(model):
    names=['cls_head','reg_head']
    if getattr(model,'use_dir',False):names.append('dir_head')
    heads=nn.ModuleDict({name:getattr(model,name) for name in names})
    if any(not isinstance(head,nn.Conv2d) or head.kernel_size!=(1,1) for head in heads.values()):
        raise ValueError('Expected existing 1x1 detection heads')
    return heads


def enable_only_detection_heads(model):
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    heads=detection_heads(model)
    for parameter in heads.parameters():parameter.requires_grad_(True)
    return heads


def spatial_head_checkpoint(model,contract,epoch):
    return dict(format='pointwise_stage_and_detection_heads_v1',
        stage=model.backbone.resnet.layer0.state_dict(),heads=detection_heads(model).state_dict(),
        contract=contract,epoch=epoch)


def load_spatial_head_checkpoint(model,saved):
    if saved.get('format')!='pointwise_stage_and_detection_heads_v1':
        raise ValueError('Expected both pointwise stage and calibrated heads')
    model.backbone.resnet.layer0.load_state_dict(saved['stage'],strict=True)
    detection_heads(model).load_state_dict(saved['heads'],strict=True)

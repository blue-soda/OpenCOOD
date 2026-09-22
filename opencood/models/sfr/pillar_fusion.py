"""Transport encoded pillars before BEV decoding, preserving the frozen single fallback."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import splat


class SfrPillarFusion(nn.Module):
    def __init__(self,sender,settings):
        super().__init__()
        self.bounds=sender.bounds
        self.orientation_conditioned=settings.get('orientation_conditioned',False)
        c=64
        self.adapter=nn.Conv2d(c+2*int(self.orientation_conditioned),c,1)
        with torch.no_grad():
            self.adapter.weight.zero_();self.adapter.weight[:,:c,0,0].copy_(torch.eye(c));self.adapter.bias.zero_()
        self.backbone=copy.deepcopy(sender.backbone).requires_grad_(True)
        self.shrink=copy.deepcopy(sender.shrink_conv).requires_grad_(True) if sender.shrink_flag else nn.Identity()
        self.cls_head=copy.deepcopy(sender.cls_head).requires_grad_(True)
        self.reg_head=copy.deepcopy(sender.reg_head).requires_grad_(True)
        self.dir_head=copy.deepcopy(sender.dir_head).requires_grad_(True) if sender.use_dir else None
        self.single_backbone=copy.deepcopy(sender.backbone).requires_grad_(False)
        self.single_shrink=copy.deepcopy(sender.shrink_conv).requires_grad_(False) if sender.shrink_flag else nn.Identity()
        self.single_cls_head=copy.deepcopy(sender.cls_head).requires_grad_(False)
        self.single_reg_head=copy.deepcopy(sender.reg_head).requires_grad_(False)
        self.single_dir_head=copy.deepcopy(sender.dir_head).requires_grad_(False) if sender.use_dir else None

    def train(self,mode=True):
        super().train(mode)
        for name,module in self.named_children():
            if name.startswith('single_'):module.eval()
        return self

    def forward(self,ego,transported,return_components=False):
        with torch.no_grad():
            single=self.single_shrink(self.single_backbone(dict(spatial_features=ego))['spatial_features_2d'])
            baseline=dict(psm=self.single_cls_head(single),rm=self.single_reg_head(single))
            if self.single_dir_head is not None:baseline['dm']=self.single_dir_head(single)
        mass=ego.new_zeros(1,1,*ego.shape[-2:])
        if transported is None:
            diagnostics=dict(coverage_fraction=0.,coverage_mass=0.)
            if return_components:
                diagnostics['components']=(baseline,baseline,ego.new_zeros(1,1,*baseline['psm'].shape[-2:]))
            return baseline,diagnostics
        xyz,features,ages,scores=transported[:4]
        inputs=torch.cat((features,transported[4]),-1) if self.orientation_conditioned else features
        received,mass=splat(xyz,inputs,self.bounds,ego.shape[-2:],scores)
        support=mass.clamp(0,1)
        adapted=self.adapter(received)
        fused=ego+support*(torch.maximum(ego,adapted)-ego)
        decoded=self.shrink(self.backbone(dict(spatial_features=fused))['spatial_features_2d'])
        output=dict(psm=self.cls_head(decoded),rm=self.reg_head(decoded))
        if self.dir_head is not None:output['dm']=self.dir_head(decoded)
        # Decode neighbourhood around received foreground rather than only the
        # exact sampled cell; no support means exact single predictions.
        mask=F.adaptive_max_pool2d(F.max_pool2d(support,7,stride=1,padding=3),single.shape[-2:])
        decoded_output=output
        output={key:baseline[key]+mask*(value-baseline[key]) for key,value in output.items()}
        diagnostics=dict(coverage_fraction=float((support>0).float().mean()),coverage_mass=float(support.sum()),
                         prediction_support_fraction=float((mask>0).float().mean()))
        if return_components:diagnostics['components']=(baseline,decoded_output,mask)
        return output,diagnostics

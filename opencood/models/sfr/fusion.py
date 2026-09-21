"""Final BEV fusion with strict empty-collaboration fallback and standard heads."""
import copy
import torch
from torch import nn
from .geometry import splat


class SfrFusion(nn.Module):
    def __init__(self, sender, settings):
        super().__init__()
        c, h = sender.out_channel, settings['hidden_dim']
        self.bounds = sender.bounds
        self.adapter = nn.Conv2d(c, c, 1)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, :, 0, 0].copy_(torch.eye(c))
            self.adapter.bias.zero_()
        self.residual = nn.Sequential(nn.Conv2d(2*c+2, h, 1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, c, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.cls_head, self.reg_head = copy.deepcopy(sender.cls_head), copy.deepcopy(sender.reg_head)
        self.dir_head = copy.deepcopy(sender.dir_head) if sender.use_dir else None
        self.adapter.requires_grad_(True)
        self.residual.requires_grad_(True)
        for head in (self.cls_head, self.reg_head, self.dir_head):
            if head is not None:
                head.requires_grad_(settings['train_heads'])

    def forward(self, ego, transported):
        if transported is None:
            fused = ego
            coverage = ego.new_zeros(1, 1, *ego.shape[-2:])
        else:
            xyz, features, ages, scores = transported
            combined = torch.cat((features, ages[:, None]), -1)
            received, mass = splat(xyz, combined, self.bounds, ego.shape[-2:], scores)
            collab, age = received[:, :-1], received[:, -1:]
            coverage = mass.clamp(0, 1)
            adapted = self.adapter(collab)
            base = torch.maximum(ego, adapted)
            residual = self.residual(torch.cat((ego, adapted, coverage, age), 1))
            fused = ego+coverage*(base-ego+residual)
        output = dict(psm=self.cls_head(fused), rm=self.reg_head(fused))
        if self.dir_head is not None:
            output['dm'] = self.dir_head(fused)
        return output, {'coverage_fraction': float((coverage > 0).float().mean()), 'coverage_mass': float(coverage.sum())}

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
        self.orientation_conditioned = settings.get('orientation_conditioned', False)
        self.train_heads = settings['train_heads']
        initialization_rng = torch.random.get_rng_state()
        self.adapter = nn.Conv2d(c+2*int(self.orientation_conditioned), c, 1)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, :c, 0, 0].copy_(torch.eye(c))
            self.adapter.bias.zero_()
        # Adapter initialization is overwritten; keep common residual weights
        # identical under the same seed in orientation-on/off comparisons.
        torch.random.set_rng_state(initialization_rng)
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
        if self.train_heads:
            self.single_cls_head = copy.deepcopy(sender.cls_head).requires_grad_(False)
            self.single_reg_head = copy.deepcopy(sender.reg_head).requires_grad_(False)
            self.single_dir_head = copy.deepcopy(sender.dir_head).requires_grad_(False) if sender.use_dir else None

    def forward(self, ego, transported):
        if transported is None:
            fused = ego
            coverage = ego.new_zeros(1, 1, *ego.shape[-2:])
        else:
            xyz, features, ages, scores = transported[:4]
            extra = (transported[4],) if self.orientation_conditioned else ()
            combined = torch.cat((features, ages[:, None])+extra, -1)
            received, mass = splat(xyz, combined, self.bounds, ego.shape[-2:], scores)
            c = features.shape[-1]
            collab, age = received[:, :c], received[:, c:c+1]
            coverage = mass.clamp(0, 1)
            adapter_input = torch.cat((collab, received[:, c+1:]), 1) if self.orientation_conditioned else collab
            adapted = self.adapter(adapter_input)
            base = torch.maximum(ego, adapted)
            residual = self.residual(torch.cat((ego, adapted, coverage, age), 1))
            fused = ego+coverage*(base-ego+residual)
        output = dict(psm=self.cls_head(fused), rm=self.reg_head(fused))
        if self.dir_head is not None:
            output['dm'] = self.dir_head(fused)
        if self.train_heads:
            # Detection adaptation is confined to received support. Empty or
            # out-of-range collaboration remains exactly the frozen single model.
            single = dict(psm=self.single_cls_head(ego), rm=self.single_reg_head(ego))
            if self.single_dir_head is not None: single['dm'] = self.single_dir_head(ego)
            output = {key: single[key]+coverage*(value-single[key]) for key, value in output.items()}
        return output, {'coverage_fraction': float((coverage > 0).float().mean()), 'coverage_mass': float(coverage.sum())}

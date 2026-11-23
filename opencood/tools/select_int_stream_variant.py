"""Fit single-interface variants and score causal, weighted-event fusion and projection reuse."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.audit_int_spike_pyramid_ops import ConvLedger
from opencood.tools.probe_int_local_ann_interfaces import ReusedProjection
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic


class Interface(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike = Multispike()
        self.collect = True
        self.squares = self.counts = None

    def forward(self, x):
        if self.collect:
            axes = (0, 2, 3)
            squares, counts = x.square().sum(axes), (x > 0).sum(axes)
            self.squares = squares if self.squares is None else self.squares + squares
            self.counts = counts if self.counts is None else self.counts + counts
            return x
        q = self.log_step.exp().clamp(1e-5, 100.).view(1, -1, 1, 1)
        return self.spike(x / q) * q

    def calibrated(self):
        step = .75 * (self.squares / self.counts.clamp_min(1)).sqrt()
        self.log_step = nn.Parameter(step.clamp_min(1e-5).log())
        self.collect = False


class FusionCost:
    """Compare matched static and temporal tap counts, including border and refresh."""
    def __init__(self, physical, static_only):
        self.physical = physical
        self.static_only = static_only
        self.history = {}
        self.taps = {}
        self.static = self.delta = self.chosen = self.dense = 0
        self.bookkeeping = self.storage = 0
        self.max_error = 0.

    def hook(self, layer, args, reference):
        x = args[0]
        assert x.shape[0] == layer.groups == 1 and layer.stride == layer.dilation == (1, 1)
        assert x.shape[-2:] == reference.shape[-2:]
        k = layer.kernel_size
        if layer not in self.taps:
            ones = torch.ones_like(x[:, :1])
            self.taps[layer] = F.conv2d(ones, x.new_ones(1, 1, *k), padding=layer.padding)
        taps = self.taps[layer]
        static = int(((x != 0).sum(1) * taps).sum(dtype=torch.float64)) * layer.out_channels
        old = self.history.get(layer)
        change = x if old is None else x - old[0]
        delta = static if old is None else int(
            ((change != 0).sum(1) * taps).sum(dtype=torch.float64)) * layer.out_channels
        # The two choices have the same spatial grid. Include subtract/add in the decision.
        reuse = old is not None and 2 * delta + reference.numel() < 2 * static
        self.dense += reference.numel() * layer.in_channels * k[0] * k[1]
        self.static += static
        self.delta += delta
        self.chosen += delta if reuse else static
        if old is not None:
            self.bookkeeping += x.numel()
        if reuse:
            self.bookkeeping += reference.numel()
        output = reference
        if self.physical:
            z = change if reuse else x
            patches = F.unfold(z, k, padding=layer.padding)[0].T.contiguous().to_sparse_csr()
            flat = torch.sparse.mm(patches, layer.weight.flatten(1).T.contiguous())
            output = flat.T.reshape_as(reference)
            if reuse:
                output = output + old[1]
            elif layer.bias is not None:
                output = output + layer.bias.view(1, -1, 1, 1)
            err = (output - reference).abs().max().item()
            self.max_error = max(self.max_error, err)
            if not torch.allclose(output, reference, rtol=2e-4, atol=5e-4):
                raise ValueError(('fusion reconstruction', err, k))
        if not self.static_only:
            self.history[layer] = (x.detach().clone(), output.detach().clone())
        return output


def relocated(rows, plan, dataset):
    verified = set()
    for row in rows:
        for field in ('pcd', 'label'):
            stored = row[field]
            if not stored:
                continue
            path = dataset / Path(stored).relative_to('/home/fzhh/gzc/dataset')
            if stored not in verified:
                assert fit.digest(path) == plan['members'][stored], path
                verified.add(stored)
            row[field] = str(path)
    return rows


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    receipt = fit.read(a.root / 'split/report.json')
    plan_path = a.root / 'split/plan.json'
    assert fit.digest(plan_path) == receipt['plan']
    plan = fit.read(plan_path)
    scenes = receipt['splits']['fit']['scenes']
    train = relocated([r for r in plan['rows']['fit'] if r['scene'] in scenes[:4]], plan, a.dataset)
    monitor = periodic(relocated([r for r in plan['rows']['fit'] if r['scene'] in scenes[-2:]], plan, a.dataset))
    evaluation = monitor
    if a.internal:
        evaluation = periodic(relocated(plan['rows']['internal'], plan, a.dataset))
        assert not set(r['scene'] for r in evaluation).intersection(scenes)
    if a.limit:
        evaluation = evaluation[:a.limit]
    init = a.root / 'initial.pt'
    assert fit.digest(init) == '57f9d146f2a4dd1344ad36dd670bc7f9bcdfd85ae04aaa8c4855cc635b148c33'
    ck = a.root / ('fine_temporal_%s.pt' % a.seed)
    assert fit.digest(ck) == {303:'e8ae1933aaa9da4103f0dab1cc0eca89eaa08d6e3bf43fe35eeea20c59f3352b',
                              707:'dc11c3232a40781da9e475ace6dc1cba6576c82c520997f2449c9bf9f504c56c'}[a.seed]
    cfg = fit.load_yaml(str(a.root / 'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(a.root / 'inputs', cfg)
    net = RecoveryPyramid(source.core, init, 'fine_temporal').cuda()
    net.load_state_dict(torch.load(ck, map_location='cpu')['net'])
    criterion = fit.PointPillarDirLoss(cfg['loss']['args']).cuda()
    bins = {}
    def replay(rows, opt=None, epoch=0):
        with torch.set_grad_enabled(opt is not None):
            return fit.replay(rows, cfg, source, net, criterion, opt, epoch, a.out, bins,
                              order_seed=a.seed, window_mode='fixed')
    net.interfaces = nn.ModuleList()
    def attach(module):
        interface = Interface().cuda()
        net.interfaces.append(interface)
        def apply_interface(_, args):
            return (interface(args[0]),) + args[1:]
        module.register_forward_pre_hook(apply_interface)
    if a.arm == 'pillar':
        attach(net.input_sender)  # Same positive quantization as the tested pre-pooling PFN.
    if a.arm == 'up':
        for branch in source.core.deblocks:
            attach(branch)
    if a.arm == 'fusion':
        attach(source.core.readout)
    if len(net.interfaces):
        replay(periodic(train[:20]))
        for interface in net.interfaces:
            interface.calibrated()
    if a.weights:
        net.load_state_dict(torch.load(a.weights, map_location='cpu'))
    if a.train:
        params = [(k,p) for k,p in net.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([
            dict(params=[p for k,p in params if not k.startswith('interfaces.')], lr=1e-5),
            dict(params=list(net.interfaces.parameters()), lr=1e-3)], weight_decay=1e-4)
        for epoch in (1, 2):
            training = replay(train, opt, epoch)
            assert training['updates'] == 34
            fit.write_json(a.out / ('train_%d.json' % epoch), training)
        torch.save(net.state_dict(), a.out / 'trained.pt')
    net.eval()
    ledger = ConvLedger(net)
    fusion = FusionCost(a.physical, a.static_fusion)
    projections = []
    for branch in source.core.deblocks:
        branch[0] = ReusedProjection(branch[0])
        projections.append(branch[0])
    linear_ops = [0, 0]
    def pillar_cost(layer, args, out):
        linear_ops[0] += out.numel() * layer.in_features * 2
    def head_cost(layer, args, out):
        linear_ops[1] += out.numel() * layer.in_channels * 2
    for module in source.pillar_vfe.modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(pillar_cost)
    for layer in net.heads.values():
        layer.register_forward_hook(head_cost)
    frame = [0]
    def reset(module, args):
        if evaluation[frame[0]]['reset_before']:
            fusion.history.clear()
            for proj in projections:
                proj.old = proj.output = None
        frame[0] += 1
    net.register_forward_pre_hook(reset)
    for module in source.core.readout.modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(fusion.hook)
    fit.delta_conv = ledger.count
    try:
        result = replay(evaluation)
    finally:
        fit.delta_conv = ledger.original
    n = frame[0]
    assert n == len(evaluation)
    other = sum(linear_ops) + 2 * sum(float(v[1]) for v in ledger.totals.values())
    up = 2 * sum(p.temporal_mac for p in projections)
    up_static = 2 * sum(p.static_mac for p in projections)
    cost = lambda value: value / n / 1e9
    report = dict(metrics=result, seed=a.seed, arm=a.arm, physical_fusion=a.physical, static_fusion=a.static_fusion,
                  full_cost_g=cost(other+up+2*fusion.chosen+fusion.bookkeeping),
                  temporal_up_static_fusion_g=cost(other+up+2*fusion.static),
                  static_cost_g=cost(other+up_static+2*fusion.static),
                  fusion_dense_g=cost(2*fusion.dense), fusion_static_g=cost(2*fusion.static),
                  fusion_chosen_g=cost(2*fusion.chosen), fusion_state_arithmetic_g=cost(fusion.bookkeeping),
                  up_g=cost(up), up_static_g=cost(up_static), other_g=cost(other),
                  fusion_dense_reference_max_error=fusion.max_error if a.physical else None,
                  fusion_state_bytes=sum(x.numel()*x.element_size() for pair in fusion.history.values() for x in pair),
                  scope='linear and convolution arithmetic plus fusion delta formation and accumulation; support selection, quantization, memory traffic and upsample state ops excluded')
    fit.write_json(a.out / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('arm', choices=('parent', 'up', 'pillar', 'fusion'))
    parser.add_argument('--seed', type=int, choices=(303,707), default=303)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--internal', action='store_true')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--static-fusion', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--weights', type=Path)
    run(parser.parse_args())

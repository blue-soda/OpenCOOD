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
from opencood.tools.train_int_full_dair_pair import full_rows
from opencood.data_utils.int_v2v_ego_stream import v2v_rows


class Interface(nn.Module):
    def __init__(self, residual_scale=None):
        super().__init__()
        self.spike = Multispike()
        self.collect = True
        self.squares = self.counts = None
        self.residual_scale = residual_scale
        self.previous = None
        self.record = False
        self.count_amplitude = False
        self.elements = self.escapes = self.fired = 0
        self.continuous = None
        self.shadow = None

    def forward(self, x):
        if self.collect:
            axes = (0, 2, 3)
            squares, counts = x.square().sum(axes), (x > 0).sum(axes)
            self.squares = squares if self.squares is None else self.squares + squares
            self.counts = counts if self.counts is None else self.counts + counts
            return x
        q = self.log_step.exp().clamp(1e-5, 100.).view(1, -1, 1, 1)
        if self.residual_scale is not None:
            if self.previous is None:
                self.previous = x.detach()
                if self.shadow is not None:
                    for entry in self.shadow.values():
                        entry[0] = x.detach()
                return x
            delta = x - self.previous
            code = delta.abs() / q
            escape = code > 4.5
            if self.continuous is None:
                quant = self.spike(code) * delta.sign() * q
                value = torch.where(escape, x, self.previous + quant)
            else:
                value = torch.where(code > .5 * self.continuous, x, self.previous)
            if self.shadow is not None:
                for ratio, entry in self.shadow.items():
                    if ratio == 0.:
                        entry[1] += int(torch.count_nonzero(value != self.previous))
                    else:
                        send = (x-entry[0]).abs() > (.5 * ratio * q)
                        entry[1] += int(torch.count_nonzero(send))
                        entry[0] = torch.where(send, x, entry[0]).detach()
            if self.record:
                self.elements += x.numel()
                self.fired += int(torch.count_nonzero(value != self.previous))
                self.escapes += int(torch.count_nonzero(escape))
            self.previous = value.detach()
            return value
        if self.record and self.count_amplitude:
            self.elements += x.numel()
        return self.spike(x / q) * q

    def calibrated(self):
        step = .75 * (self.squares / self.counts.clamp_min(1)).sqrt()
        if self.residual_scale is not None:
            step = step * self.residual_scale
        self.log_step = nn.Parameter(step.clamp_min(1e-5).log())
        self.collect = False


def install_pillar_up(net, core, saved=None, bypass=None):
    """Use the same four positive Multispike interfaces for training and scoring."""
    net.pillar_up = nn.ModuleList()
    for index, target in enumerate([net.input_sender, *core.deblocks]):
        interface = Interface().cuda()
        interface.count_amplitude = True
        if saved is not None:
            value = saved['pillar_up.%d.log_step' % index]
            interface.log_step = nn.Parameter(value.detach().clone().cuda())
            interface.collect = False
        net.pillar_up.append(interface)
        if (bypass == 'pillar' and index == 0) or (bypass == 'up' and index > 0):
            continue

        def quantize(_, args, sender=interface):
            return (sender(args[0]),) + args[1:]
        target.register_forward_pre_hook(quantize)
    assert len(net.pillar_up) == 4


def sparse_fusion_conv(layer, z):
    """Apply just the nonzero weighted products; the caller owns bias and history."""
    patches = F.unfold(z, layer.kernel_size, padding=layer.padding)[0].T.contiguous().to_sparse_csr()
    flat = torch.sparse.mm(patches, layer.weight.flatten(1).T.contiguous())
    return flat.T.reshape(1, layer.out_channels, *z.shape[-2:])


class FusionCost:
    """Compare matched static and temporal tap counts, including border and refresh."""
    def __init__(self, physical, static_only, first_only):
        self.physical = physical
        self.static_only = static_only
        self.first_only = first_only
        self.first = None
        self.history = {}
        self.taps = {}
        self.static = self.delta = self.chosen = self.dense = 0
        self.bookkeeping = self.storage = 0
        self.max_error = 0.

    def project(self, layer, x, reference=None):
        assert x.shape[0] == layer.groups == 1 and layer.stride == layer.dilation == (1, 1)
        height, width = x.shape[-2:]
        output_elements = height * width * layer.out_channels
        k = layer.kernel_size
        if self.first is None:
            self.first = layer
        remember = not self.static_only and (not self.first_only or layer is self.first)
        assert tuple(2*p+1 for p in layer.padding) == k
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
        reuse = old is not None and 2 * delta + output_elements < 2 * static
        self.dense += output_elements * layer.in_channels * k[0] * k[1]
        self.static += static
        self.delta += delta
        self.chosen += delta if reuse else static
        if old is not None:
            self.bookkeeping += x.numel()
        if reuse:
            self.bookkeeping += output_elements
        output = reference
        if self.physical:
            z = change if reuse else x
            output = sparse_fusion_conv(layer, z)
            if reuse:
                output = output + old[1]
            elif layer.bias is not None:
                output = output + layer.bias.view(1, -1, 1, 1)
            if reference is not None:
                err = (output - reference).abs().max().item()
                self.max_error = max(self.max_error, err)
                if not torch.allclose(output, reference, rtol=2e-4, atol=5e-4):
                    raise ValueError(('fusion reconstruction', err, k))
        if remember:
            # The first fusion convolution only reads the concatenated feature.
            # Keep its existing storage, including a sender-owned corrected level.
            stored = x.detach() if layer is self.first else x.detach().clone()
            self.history[layer] = (stored, output.detach().clone())
        return output

    def hook(self, layer, args, reference):
        return self.project(layer, args[0], reference)


class SparseFusionLayer(nn.Module):
    """A scoring path that never evaluates the original dense fusion convolution."""
    def __init__(self, layer, fusion):
        super().__init__()
        self.layer, self.fusion = layer, fusion

    def forward(self, x):
        return self.fusion.project(self.layer, x)


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
    if a.full_val:
        evaluation = periodic(full_rows(a.root / 'sequence_manifest.json', a.dataset, 'val'))
        assert len(evaluation) == 5794
    if a.v2v_split:
        manifest = a.root / 'formal_splits/v2v_official_deduplicated.jsonl'
        target_train = v2v_rows(manifest, 'train')
        scenes = list(dict.fromkeys(r['scene'] for r in target_train))
        train = [r for r in target_train if r['scene'] in scenes[:4]]
        evaluation = periodic(v2v_rows(manifest, a.v2v_split))
    if a.limit:
        evaluation = evaluation[:a.limit]
    init = a.root / 'initial.pt'
    assert fit.digest(init) == '57f9d146f2a4dd1344ad36dd670bc7f9bcdfd85ae04aaa8c4855cc635b148c33'
    ck = a.root / ('fine_temporal_%s.pt' % a.seed)
    cfg = fit.load_yaml(str(a.root / 'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(a.root / 'inputs', cfg)
    net = RecoveryPyramid(source.core, init, 'fine_temporal').cuda()
    if a.full_parent:
        full = torch.load(a.full_parent, map_location='cpu')
        assert full['contract']['seed'] == a.seed
        assert full['contract']['labels_per_epoch'] == (6958 if a.v2v_split else 4678)
        trained_variant = full['contract']['variant']
        assert trained_variant in ('fine_temporal', 'fine_independent')
        if trained_variant == 'fine_independent' and not a.clear:
            raise ValueError('A per-scan training control must also clear history at evaluation.')
        if a.bypass_joint and not full['contract'].get('pillar_up'):
            raise ValueError('Interface bypass requires the jointly trained checkpoint.')
        if full['contract'].get('pillar_up'):
            install_pillar_up(net, source.core, full['net'], a.bypass_joint)
        if 'continuous_threshold' in full['contract']:
            if not a.continuous_fusion:
                raise ValueError('Use continuous fusion when scoring the full continuous control.')
            for sender in net.senders:
                sender.continuous_threshold = full['contract']['continuous_threshold']
        net.load_state_dict(full['net'])
        ck = a.full_parent
    else:
        if a.bypass_joint:
            raise ValueError('Interface bypass requires --full-parent.')
        assert fit.digest(ck) == {303:'e8ae1933aaa9da4103f0dab1cc0eca89eaa08d6e3bf43fe35eeea20c59f3352b',
                                  707:'dc11c3232a40781da9e475ace6dc1cba6576c82c520997f2449c9bf9f504c56c'}[a.seed]
        net.load_state_dict(torch.load(ck,map_location='cpu')['net'])
    criterion = fit.PointPillarDirLoss(cfg['loss']['args']).cuda()
    bins = {}
    def replay(rows, opt=None, epoch=0, record=False):
        with torch.set_grad_enabled(opt is not None):
            return fit.replay(rows, cfg, source, net, criterion, opt, epoch, a.out, bins,
                              order_seed=a.seed, window_mode='fixed',
                              memoryless=record and a.clear,
                              frame_log=a.out / 'frames.jsonl' if record and a.frames else None,
                              target_log=a.out / 'targets.jsonl' if record and a.target_records else None)
    net.interfaces = nn.ModuleList()
    def attach(module, residual_scale=None):
        interface = Interface(residual_scale).cuda()
        net.interfaces.append(interface)
        def apply_interface(_, args):
            return (interface(args[0]),) + args[1:]
        module.register_forward_pre_hook(apply_interface)
        if residual_scale is not None:
            def reset_sender(_, args):
                if args[2] is None:
                    interface.previous = None
            net.register_forward_pre_hook(reset_sender)
    if a.arm == 'pillar':
        attach(net.input_sender)  # Same positive quantization as the tested pre-pooling PFN.
    if a.arm == 'up':
        for branch in source.core.deblocks:
            attach(branch)
    if a.arm == 'fusion':
        attach(source.core.readout)
    if a.arm in ('delta_fine', 'delta_medium'):
        attach(source.core.readout, .125 if a.arm == 'delta_fine' else .25)
    if len(net.interfaces):
        if a.scales:
            frozen = fit.read(a.scales)
            assert (frozen['seed'], frozen['arm'], frozen['parent_sha']) == (a.seed, a.arm, fit.digest(ck))
            assert len(frozen['log_step']) == len(net.interfaces)
            for interface, values in zip(net.interfaces, frozen['log_step']):
                interface.log_step = nn.Parameter(torch.tensor(values, device='cuda', dtype=torch.float32))
                interface.collect = False
        else:
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
    if a.continuous_fusion:
        if a.train or not a.arm.startswith('delta_'):
            raise ValueError('Matched continuous control expects a fixed residual sender.')
        interface, = net.interfaces
        ratios = (.75, .875, 1., 1.125, 1.25)
        interface.shadow = {ratio:[None, 0] for ratio in (0.,) + ratios}
        replay(periodic(train))
        visits = {ratio:entry[1] for ratio,entry in interface.shadow.items()}
        interface.continuous = min(ratios, key=lambda ratio:abs(visits[ratio] - visits[0.]))
        fit.write_json(a.out / 'continuous_calibration.json', dict(
            train_scenes=scenes[:4], transmitted_elements=visits, selected_ratio=interface.continuous))
        interface.shadow = None
        interface.previous = None
    if len(net.interfaces):
        fit.write_json(a.out / 'scales.json', dict(seed=a.seed, arm=a.arm, parent_sha=fit.digest(ck),
            plan_sha=receipt['plan'], log_step=[i.log_step.detach().cpu().tolist() for i in net.interfaces]))
    for interface in net.interfaces:
        interface.record = True
    joint = getattr(net, 'pillar_up', ())
    for interface in joint:
        interface.record = True
    ledger = ConvLedger(net)
    fusion = FusionCost(a.physical, a.static_fusion, a.first_fusion)
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
        if a.clear or evaluation[frame[0]]['reset_before']:
            fusion.history.clear()
            for proj in projections:
                proj.old = proj.output = None
        frame[0] += 1
    net.register_forward_pre_hook(reset)
    if a.no_reference and not a.physical:
        raise ValueError('--no-reference requires --physical')
    for parent in list(source.core.readout.modules()):
        for name, module in list(parent.named_children()):
            if isinstance(module, nn.Conv2d):
                if a.no_reference:
                    setattr(parent, name, SparseFusionLayer(module, fusion))
                else:
                    module.register_forward_hook(fusion.hook)
    fit.delta_conv = ledger.count
    try:
        result = replay(evaluation, record=True)
    finally:
        fit.delta_conv = ledger.original
    n = frame[0]
    assert n == len(evaluation)
    other = sum(linear_ops) + 2 * sum(float(v[1]) for v in ledger.totals.values())
    up = 2 * sum(p.temporal_mac for p in projections)
    up_static = 2 * sum(p.static_mac for p in projections)
    cost = lambda value: value / n / 1e9
    encoded = sum(i.elements for i in net.interfaces)
    # Ten scalar arithmetic/comparison operations per non-anchor input, conservative
    # uniform charge. No equivalent hardware-energy assumption is made here.
    encoding_ops = (4 if a.continuous_fusion else 10) * encoded
    # Divide, clamp on both sides, offset, floor, and rescale. Count separately
    # from bounded fusion, including zero BEV cells actually visited by this implementation.
    joint_encoding_ops = 6 * sum(i.elements for i in joint)
    saved = [x for pair in fusion.history.values() for x in pair]
    saved.extend(i.previous for i in net.interfaces if i.previous is not None)
    unique = {x.storage().data_ptr(): x.storage().size() * x.element_size() for x in saved}
    report = dict(metrics=result, seed=a.seed, arm=a.arm, clear_history=a.clear,
                  continuous_backbone=net.input_sender.continuous_threshold is not None,
                  bypass_joint=a.bypass_joint,
                  parent_sha=fit.digest(ck), parent_variant=trained_variant if a.full_parent else 'fine_temporal',
                  continuous_fusion=a.continuous_fusion,
                  physical_fusion=a.physical, static_fusion=a.static_fusion, dense_fusion_reference=not a.no_reference,
                  first_fusion=a.first_fusion, unique_sender_and_fusion_bytes=sum(unique.values()),
                  full_cost_g=cost(other+up+2*fusion.chosen+fusion.bookkeeping+encoding_ops+joint_encoding_ops),
                  temporal_up_static_fusion_g=cost(other+up+2*fusion.static+encoding_ops+joint_encoding_ops),
                  static_cost_g=cost(other+up_static+2*fusion.static+encoding_ops+joint_encoding_ops),
                  pillar_up_encoding_g=cost(joint_encoding_ops),
                  bounded_sender_g=cost(encoding_ops), ordinary_elements=encoded,
                  ordinary_changed=sum(i.fired for i in net.interfaces),
                  ordinary_escape=sum(i.escapes for i in net.interfaces),
                  sender_state_bytes=sum(i.previous.numel()*i.previous.element_size()
                                         for i in net.interfaces if i.previous is not None),
                  fusion_dense_g=cost(2*fusion.dense), fusion_static_g=cost(2*fusion.static),
                  fusion_chosen_g=cost(2*fusion.chosen), fusion_state_arithmetic_g=cost(fusion.bookkeeping),
                  up_g=cost(up), up_static_g=cost(up_static), other_g=cost(other),
                  fusion_dense_reference_max_error=fusion.max_error if a.physical and not a.no_reference else None,
                  fusion_state_bytes=sum(x.numel()*x.element_size() for pair in fusion.history.values() for x in pair),
                  scope='linear and convolution arithmetic plus fusion delta formation and accumulation; support selection, quantization, memory traffic and upsample state ops excluded')
    fit.write_json(a.out / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('arm', choices=('parent', 'up', 'pillar', 'fusion', 'delta_fine', 'delta_medium'))
    parser.add_argument('--seed', type=int, choices=(303,707), default=303)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--internal', action='store_true')
    parser.add_argument('--full-val', action='store_true', help='Score the frozen full DAIR validation stream.')
    parser.add_argument('--v2v-split',choices=('val','test'),help='Calibrate on target training data and score the complete requested split.')
    parser.add_argument('--full-parent', type=Path, help='Weights from the matched full DAIR backbone fit.')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--no-reference', action='store_true', help='Run sparse fusion without the verification convolution.')
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--static-fusion', action='store_true')
    policy.add_argument('--first-fusion', action='store_true', help='Retain only the linear result touched directly by the fusion sender.')
    parser.add_argument('--frames', action='store_true')
    parser.add_argument('--target-records', action='store_true', help='Record per-object scored V2V validation matches.')
    parser.add_argument('--clear', action='store_true', help='Reset task and projection history on every scan at evaluation.')
    parser.add_argument('--continuous-fusion', action='store_true',
                        help='Calibrate a same-history continuous deadband on training traffic alone.')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--bypass-joint', choices=('pillar','up'),
                        help='For a frozen jointly trained checkpoint, bypass one family of input encoders.')
    parser.add_argument('--scales', type=Path, help='Reuse the fixed interface calibration for this parent.')
    run(parser.parse_args())

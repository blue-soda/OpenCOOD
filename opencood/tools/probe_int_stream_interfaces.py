"""Frozen DAIR probe for the Pillar and deblock interfaces of the temporal SNN."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.probe_int_local_ann_interfaces import ExactPillars, ReusedProjection
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic


class InterfaceProbe:
    def __init__(self):
        self.spike = Multispike()
        self.channel_stats = {}
        self.steps = {}

    def start(self, arm):
        self.arm = arm
        self.pillars = None
        self.projected = {}
        self.scans = 0
        self.exact = self.overlap = self.total = 0
        self.mac = [0, 0, 0]
        self.max_error = 0.

    def encode(self, name, x):
        flat = x.movedim(1, -1).reshape(-1, x.shape[1])
        if self.arm == 'calibrate':
            values = flat.square().sum(0), (flat > 0).sum(0)
            if name in self.channel_stats:
                values = tuple(a + b for a, b in zip(self.channel_stats[name], values))
            self.channel_stats[name] = values
        if self.arm == name.split('_')[0] or self.arm == 'both' or (
                self.arm == 'pointwise_both' and name.startswith('upsample_')):
            step = self.steps[name].view(1, -1, *([1] * (x.ndim - 2)))
            return self.spike(x / step) * step
        return x

    def pillar(self, module, args, output):
        batch = args[0]
        if self.arm == 'reference':
            coords = batch['voxel_coords'].cpu().numpy()
            points = batch['voxel_features'].cpu().numpy()
            counts = batch['voxel_num_points'].cpu().numpy()
            current = {tuple(c): (int(n), p[:n].tobytes()) for c, n, p in zip(coords, counts, points)}
            assert len(current) == len(coords)
            if self.pillars is not None and not self.rows[self.scans]['reset_before']:
                self.overlap += len(self.pillars.keys() & current.keys())
                self.exact += sum(self.pillars.get(c) == value for c, value in current.items())
            self.total += len(current)
            self.pillars = current
        self.scans += 1
        output['pillar_features'] = self.encode('pillar', output['pillar_features'])
        return output

    def upsample(self, name):
        def hook(module, args):
            return (self.encode(name, args[0]),)
        return hook

    def audit_projection(self, module, args, output):
        if self.arm != 'reference':
            return
        x = args[0]
        reset = self.rows[self.scans - 1]['reset_before']
        if reset:
            change, reconstructed = x, output
        else:
            previous, cached = self.projected[module]
            change = x - previous
            update = F.conv_transpose2d(change, module.weight, None, module.stride,
                                        module.padding, module.output_padding, module.groups, module.dilation)
            reconstructed = cached + update
            self.max_error = max(self.max_error, (output - reconstructed).abs().max().item())
        self.projected[module] = x.clone(), reconstructed.clone()
        fanout = module.out_channels * int(np.prod(module.kernel_size))
        self.mac[0] += x.numel() * fanout
        self.mac[1] += torch.count_nonzero(x).item() * fanout
        self.mac[2] += torch.count_nonzero(change).item() * fanout
        return reconstructed


@torch.no_grad()
def run(base, output, sparse_reuse=False, pointwise=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(303)
    inputs = base / 'membrane_20261121/inputs'
    plan = base / 'silent_task_20261122/expanded_split/plan.json'
    receipt = fit.read(plan.parent / 'report.json')
    assert receipt['plan'] == fit.digest(plan)
    scenes = receipt['splits']['fit']['scenes']
    fit_rows = fit.read(plan)['rows']['fit']
    calibration = periodic([r for r in fit_rows if r['scene'] == scenes[0]][:20])
    monitor = periodic([r for r in fit_rows if r['scene'] in scenes[-2:]])
    assert calibration and monitor and scenes[0] not in scenes[-2:]
    cfg = fit.load_yaml(str(inputs / 'config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(inputs, cfg)
    net = RecoveryPyramid(source.core, base / 'native_stepfit_20261122/sweep/initial.pt', 'fine_temporal').cuda()
    ck = base / 'native_scale_history_20261123/fine_temporal_303/best.pt'
    assert fit.digest(ck) == '49538d3f5f06d73315368521d562bd0cf2bb328126d09dc433a5398bef4bd842'
    net.load_state_dict(torch.load(ck, map_location='cpu')['net'])
    net.eval()
    criterion = fit.PointPillarDirLoss(cfg['loss']['args']).cuda()
    if sparse_reuse:
        source.pillar_vfe = pillars = ExactPillars(source.pillar_vfe)
        caches = []
        for branch in source.core.deblocks:
            branch[0] = ReusedProjection(branch[0])
            caches.append(branch[0])
        frame = 0
        def reset_caches(module, args):
            nonlocal frame
            if monitor[frame]['reset_before']:
                pillars.previous = {}
                for cache in caches:
                    cache.old = cache.output = None
            frame += 1
        pillars.register_forward_pre_hook(reset_caches)
        result = fit.replay(monitor, cfg, source, net, criterion, None, 0, output, {}, window_mode='fixed')
        assert frame == len(monitor)
        fit.write_json(output / 'sparse.json', dict(metrics=result, exact_hits=pillars.hits,
            occupied_pillars=pillars.total, monitor_scenes=scenes[-2:], deblock_mac_dense_static_temporal=[
                [c.dense_mac, c.static_mac, c.temporal_mac] for c in caches]))
        return
    probe = InterfaceProbe()
    source.pillar_vfe.register_forward_hook(probe.pillar)
    if pointwise:
        layer, = source.pillar_vfe.pfn_layers
        assert layer.last_vfe and layer.use_norm
        def point_spikes(module, args, values):
            if probe.arm == 'pointwise_both':
                step = probe.steps['pillar'].view(1, -1, 1)
                return probe.spike(values / step) * step
        layer.norm.register_forward_hook(point_spikes)
    for idx, branch in enumerate(source.core.deblocks):
        conv, = [m for m in branch.modules() if isinstance(m, nn.ConvTranspose2d)]
        assert conv.groups == 1 and conv.kernel_size == conv.stride and conv.padding == (0, 0)
        conv.register_forward_pre_hook(probe.upsample('upsample_%d' % idx))
        conv.register_forward_hook(probe.audit_projection)
    bins = {}
    arms = ('calibrate', 'pointwise_both') if pointwise else (
        'calibrate', 'dense', 'reference', 'upsample', 'pillar', 'both')
    for arm in arms:
        probe.start(arm)
        probe.rows = calibration if arm == 'calibrate' else monitor
        metrics = fit.replay(probe.rows, cfg, source, net, criterion, None, 0, output, bins, window_mode='fixed')
        assert probe.scans == len(probe.rows)
        if arm == 'calibrate':
            for name, (squares, positives) in probe.channel_stats.items():
                probe.steps[name] = (.75 * (squares / positives.clamp_min(1)).sqrt()).clamp_min(1e-5)
        fit.write_json(output / (arm + '.json'), dict(
            metrics=metrics, calibration_scene=scenes[0], monitor_scenes=scenes[-2:],
            exact_ordered_pillars=probe.exact, occupied_overlap=probe.overlap, occupied_total=probe.total,
            deblock_mac_dense_static_temporal=probe.mac, max_abs_projection_error=probe.max_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sparse-reuse', action='store_true')
    mode.add_argument('--pointwise', action='store_true')
    args = parser.parse_args()
    run(args.diagnostics, args.output, args.sparse_reuse, args.pointwise)

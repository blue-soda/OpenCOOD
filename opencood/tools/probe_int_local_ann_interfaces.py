"""Measure extra exact reuse beyond the local ANN's existing deep and fusion caches."""
import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn

from opencood.tools import eval_int_branch_fusion as online
from opencood.tools import train_int_spike_pyramid as fit


class ReusedProjection(nn.Module):
    def __init__(self, conv):
        super().__init__()
        assert conv.bias is None and conv.kernel_size == conv.stride
        self.conv = conv
        self.old = self.output = None
        self.dense_mac = self.static_mac = self.temporal_mac = 0

    def forward(self, x):
        change = x if self.old is None else x - self.old
        delta, dense, active = online.sparse_phase_output(self.conv, change)
        output = delta if self.old is None else self.output + delta
        self.dense_mac += dense
        self.temporal_mac += active
        self.static_mac += int(torch.count_nonzero(x)) * self.conv.weight[0].numel()
        self.old, self.output = x.clone(), output.clone()
        return output


class ExactPillars(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.previous = {}
        self.values = None
        self.hits = self.total = 0

    def forward(self, batch):
        coords = batch['voxel_coords'].cpu().numpy()
        raw = batch['voxel_features'].cpu().numpy()
        sizes = batch['voxel_num_points'].cpu().numpy()
        keys = [(tuple(c), int(n), p[:n].tobytes()) for c, n, p in zip(coords, sizes, raw)]
        self.total += len(keys)
        hits, previous, misses = [], [], []
        for i, key in enumerate(keys):
            if key in self.previous:
                hits.append(i)
                previous.append(self.previous[key])
            else:
                misses.append(i)
        if not hits:
            result = self.original(batch)['pillar_features'].reshape(len(keys), -1)
        else:
            result = self.values.new_empty((len(keys), self.values.shape[1]))
            result[hits] = self.values[previous]
            if misses:
                part = dict(batch)
                for field in ('voxel_features', 'voxel_coords', 'voxel_num_points'):
                    part[field] = batch[field][misses]
                result[misses] = self.original(part)['pillar_features'].reshape(len(misses), -1)
        self.hits += len(hits)
        self.previous = {key: i for i, key in enumerate(keys)}
        self.values = result
        batch['pillar_features'] = result
        return batch


@torch.no_grad()
def run(base, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = base / 'membrane_20261121/inputs'
    plan = base / 'silent_task_20261122/expanded_split/plan.json'
    receipt = fit.read(plan.parent / 'report.json')
    assert fit.digest(plan) == receipt['plan']
    scenes = receipt['splits']['fit']['scenes'][-2:]
    rows = [r for r in fit.read(plan)['rows']['fit'] if r['scene'] in scenes]
    cfg = fit.load_yaml(str(inputs / 'config.yaml'))
    cfg['data_augment'] = []
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    for reuse in (False, True):
        source = fit.build_source(inputs, cfg)
        group = source.core.readout.layers[0].double_conv[0]
        group[0] = fusion = online.BranchFusion(group[0]).eval()
        model = online.local_model(source, indexed=True)
        layers = [branch[0] for branch in source.core.deblocks[:2]]
        projection = online.FloatProjectionAudit(source.core, live=True, verify=False)
        caches = []
        if reuse:
            source.pillar_vfe = pillars = ExactPillars(source.pillar_vfe)
            for branch, conv in zip(source.core.deblocks, layers):
                branch[0] = ReusedProjection(conv)
                caches.append(branch[0])
        ds, stream = fit.loader(rows, cfg, 0, 303)
        stats = {t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)}
        state = None
        frames = labels = 0
        for row, sample in zip(rows, stream):
            meta = sample['meta']
            assert meta['frame'] == row['frame']
            if frames == 0 or row.get('reset_before') or row['segment'] != rows[frames-1]['segment']:
                state = None
                selector = online.ObservedPolicy(model, partial(online.CachedResidualPolicy,
                                                                compiler=online.compact_group_bucket))
            reason = model.refresh_reason(state, meta)
            if reuse and reason:
                pillars.previous = {}
                for cache in caches:
                    cache.old = cache.output = None
            ego = fit.to_device(sample['ego'], torch.device('cuda'))
            x = fit.spatial_input(source, ego)
            projection.begin()
            fusion.begin(state, reason)
            selector.observation = None
            pred, state, _ = model.step_spatial(x, state, meta, requests=selector)
            state['branch_fusion'] = fusion.new
            if meta['supervised']:
                labels += 1
                gt = ds.post.generate_gt_bbx(dict(ego=ego))
                counts = fit.score(ds.post, ego, pred, gt)['ap']
                for t, values in stats.items():
                    for key in values:
                        values[key] += counts[t][key]
            frames += 1
        assert frames == len(rows)
        fit.write_json(output / ('reuse.json' if reuse else 'original.json'), dict(
            frames=frames, labels=labels, scenes=scenes, ap50=fit.calculate_ap(stats,.5)[0],
            ap70=fit.calculate_ap(stats,.7)[0], exact_hits=pillars.hits if reuse else None,
            occupied_pillars=pillars.total if reuse else None,
            fine_middle_mac_dense_static_temporal=[
                [c.dense_mac, c.static_mac, c.temporal_mac] for c in caches]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.diagnostics, args.output)

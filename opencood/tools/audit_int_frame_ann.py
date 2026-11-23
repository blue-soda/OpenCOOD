"""Score the continuous spatial reference and price dense and static-sparse execution on the same scans."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.select_int_stream_variant import relocated
from opencood.tools.train_int_full_dair_pair import full_rows


@torch.no_grad()
def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a.full_val:
        rows = full_rows(a.root/'sequence_manifest.json', a.dataset, 'val')
    else:
        split = a.root/'split/plan.json'
        assert fit.digest(split) == fit.read(a.root/'split/report.json')['plan']
        plan = fit.read(split)
        rows = relocated(plan['rows']['internal'], plan, a.dataset)
    cfg = fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(a.root/'inputs', cfg)
    core = source.core
    ops = defaultdict(lambda: [0, 0])
    tap_cache = {}
    scans = [0]

    def hook(group):
        def count(layer, args, output):
            x = args[0]
            if isinstance(layer, nn.Linear):
                dense = sparse = 2 * output.numel() * layer.in_features
            elif isinstance(layer, nn.ConvTranspose2d):
                assert layer.groups == 1 and layer.stride == layer.kernel_size
                assert layer.padding == layer.output_padding == (0, 0) and layer.dilation == (1, 1)
                scale = 2 * layer.weight[0].numel()
                dense, sparse = x.numel() * scale, int(torch.count_nonzero(x)) * scale
            else:
                assert x.shape[0] == 1
                shape = (1, 1, *x.shape[-2:])
                key = (shape, layer.kernel_size, layer.stride, layer.padding, layer.dilation)
                if key not in tap_cache:
                    kernel = x.new_ones(1, 1, *layer.kernel_size)
                    out = F.conv2d(x.new_ones(shape), kernel, stride=layer.stride,
                                   padding=layer.padding, dilation=layer.dilation)
                    tap_cache[key] = torch.nn.grad.conv2d_input(shape, kernel, torch.ones_like(out),
                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
                taps = tap_cache[key]
                scale = 2 * (layer.out_channels // layer.groups)
                dense = int(taps.sum(dtype=torch.float64)) * x.shape[1] * scale
                sparse = int((taps * x.ne(0).sum(1)).sum(dtype=torch.float64)) * scale
                if group == 'head':
                    sparse = dense  # The matched detector ledger charges dense heads.
            ops[group][0] += dense
            ops[group][1] += sparse
        return count

    def completed(*_):
        scans[0] += 1
    core.cls_head.register_forward_hook(completed)
    modules = dict(pillar=[source.pillar_vfe], residual=[core.observation, core.coarse_encoder,
                   core.deep_encoder], projection=[core.deblocks], fusion=[core.readout],
                   head=[core.cls_head, core.reg_head, core.dir_head])
    for group, parents in modules.items():
        for parent in parents:
            if parent is not None:
                for layer in parent.modules():
                    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                        layer.register_forward_hook(hook(group))
    quality = fit.continuous_reference(rows, cfg, source, include_unlabeled=True)
    assert scans[0] == len(rows) and set(ops) == set(modules)
    per_scan = {k: [n / len(rows) / 1e9 for n in v] for k, v in ops.items()}
    fit.write_json(a.out/'result.json', dict(scans=len(rows), labels=sum(r['supervised'] for r in rows),
        ap=quality, groups_g=per_scan, columns=['dense_valid_g', 'static_nonzero_g'],
        total_g=[sum(v[i] for v in per_scan.values()) for i in (0, 1)],
        scope='Original continuous ANN, same outputs for both analytic counts. Valid spatial taps, '
              'MAC counted twice, dense Pillar and heads, no inter-frame state. No hardware measurements.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'dataset', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--full-val', action='store_true')
    run(parser.parse_args())

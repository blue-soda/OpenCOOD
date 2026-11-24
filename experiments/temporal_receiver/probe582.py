"""Read actual refresh/ordinary input scales and verify phase folding on held-out features."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from hist577 import ConvLedger, RecoveryPyramid, attach_window, fit, full_rows, scene_windows, v2v_rows


@torch.no_grad()
def run(root, data, out, domain, kind):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    cfg = fit.load_yaml(str(root / 'inputs/config.yaml'))
    cfg['data_augment'] = []
    src = fit.build_source(root / 'inputs', cfg)
    path = root / ('carry_' + kind) / 'final.pt'
    net = RecoveryPyramid(src.core, root / 'initial.pt', 'fine_temporal').cuda().eval()
    net.load_state_dict(torch.load(path, map_location='cpu')['net'])
    if kind == 'qat':
        for sender in net.senders:
            sender.continuous_threshold, sender.payload_dtype = .5, 'int8'
    if domain == 'dair':
        val = full_rows(root / 'sequence_manifest.json', data, 'val')
    else:
        val = v2v_rows(root / 'formal_splits/v2v_official_deduplicated.jsonl', 'val', data)
    rows = attach_window(scene_windows(val)[:5], False)
    ledger = ConvLedger(net)
    names = {c:n for n,c in net.named_modules() if c in ledger.inputs}
    scales, ratios, errors = {}, {}, []

    def check(conv, events, first):
        reference = ledger.original(conv, events, first)
        if phase not in (0, 4):
            return reference
        name = names[conv]
        sender = ledger.inputs[conv][1]
        scale = sender.step() * (sender.cap / 127.) if kind == 'qat' else sender.step()
        if first:
            scales[name] = scale.clone()
        ratio = scale / scales[name]
        multiplier = int(ratio.flatten()[0].item())
        assert multiplier in (1, 2) and torch.all(ratio == multiplier)
        if not first:
            ratios[name] = multiplier
        codes = (events / scale).round()
        assert (events / scale - codes).abs().max() < 1e-4
        # Every possible dyadic base is algebraically equivalent in binary32.
        # Cycle nontrivial bases across the layers to exercise this identity.
        base = 2 ** (len(errors) % 6)
        folded = (conv.weight * scales[name].view(1, -1, 1, 1)) * base
        actual = F.conv2d(codes * (multiplier / base), folded,
                          conv.bias if first else None, conv.stride, conv.padding,
                          conv.dilation, conv.groups)
        err = (actual - reference).abs().max().item() / max(1.,reference.abs().max().item())
        assert err < 2e-4
        errors.append(err)
        return reference

    state = None
    fit.delta_conv = check
    try:
        for phase,(row,sample) in enumerate(zip(rows, fit.loader(rows,cfg,0,303)[1])):
            assert row['frame'] == sample['meta']['frame'] and row['window_ordinary'] == (phase != 0)
            ego = fit.to_device(sample['ego'],torch.device('cuda'))
            _,state,_ = net(fit.spatial_input(src,ego),src.core,state,ordinary=row['window_ordinary'])
    finally:
        fit.delta_conv = ledger.original
    assert len(scales) == len(ratios) == 27 and len(errors) == 54
    result = dict(domain=domain, kind=kind, parent=fit.digest(path), rows=[[r['scene'],r['frame']] for r in rows],
                  ordinary_multiplier=ratios, checked_full_feature_maps=len(errors),
                  max_scaled_error=max(errors))
    with (out / (kind + '_phase582.json')).open('x') as handle:
        json.dump(result,handle,indent=2)


if __name__ == '__main__':
    run(*map(Path,sys.argv[1:4]),*sys.argv[4:6])

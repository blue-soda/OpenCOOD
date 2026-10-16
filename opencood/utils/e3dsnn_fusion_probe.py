"""Descriptive residual diagnostics; ROI bounds are GT axis-aligned BEV envelopes."""
import hashlib
import torch


def detector_digest(model):
    """Hash all detector weights AND buffers, excluding fusion-only state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('fusion_net.') or name == 'fusion_scale':
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bn_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_buffers()):
        if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def probe_maps(ego, residual, fused, supported, activation):
    before = ego + residual
    maps = {
        'residual_abs_mean': residual.abs().mean(1),
        'changed_channel_fraction': (fused != ego).float().mean(1),
        'nonzero_residual_unchanged_fraction': ((residual != 0) & (fused == ego)).float().mean(1),
        'support': (supported[:, 0] > 0).float(),
    }
    if activation == 'count4':
        maps.update(above_cap_fraction=(before > 4).float().mean(1),
                    below_zero_fraction=(before < 0).float().mean(1),
                    output_at_cap_fraction=(fused == 4).float().mean(1))
    return maps


def summarize_probe(maps, gt, origin, step):
    if any(value.shape[0] != 1 for value in maps.values()):
        raise ValueError('Probe export requires batch size 1')
    support = maps['support'][0] > 0

    def summarize(region):
        selected = region & support
        count = int(selected.sum())
        return dict(supported_cells=count, **{
            key: float(value[0][selected].mean()) if count else None
            for key, value in maps.items() if key != 'support'})

    h, w = support.shape
    xs = torch.arange(w, device=support.device) * step[0] + origin[0]
    ys = torch.arange(h, device=support.device) * step[1] + origin[1]
    regions = []
    for box in gt:
        lower, upper = box[:, :2].min(0)[0], box[:, :2].max(0)[0]
        region = ((ys[:, None] >= lower[1]) & (ys[:, None] <= upper[1]) &
                  (xs[None, :] >= lower[0]) & (xs[None, :] <= upper[0]))
        regions.append(summarize(region))
    return dict(frame=summarize(torch.ones_like(support)), gt_regions=regions,
                scope='Supported BEV cells only; channel fractions include zero ego channels. '
                      'GT order matches exported boxes; axis-aligned GT envelopes at feature cell centers. '
                      'Small boxes may contain no supported centers; null means no observation.')

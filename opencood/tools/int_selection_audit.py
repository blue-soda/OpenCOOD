"""Diagnostic replay of independently recurrent selected and dense backends.

This does not relax validation assertions or inject control state into the
evaluated trajectory. Shared-input replay is discarded after the audit.
"""
import torch


def capture_step(mode, ego, state, meta):
    stage = mode.spatial.stage
    original = stage.selected_from_stem
    captured = []
    def wrapped(current, maps, mask, *args, **kwargs):
        captured.append(mask.detach().clone())
        return original(current, maps, mask, *args, **kwargs)
    stage.selected_from_stem = wrapped
    try:
        pred, new, info = mode.step(ego, state, meta)
    finally:
        stage.selected_from_stem = original
    assert len(captured) == (0 if info['spatial_refresh'] else 1)
    mask = captured[0] if captured else torch.ones_like(new['spatial']['maps'][0][:,:1], dtype=torch.bool)
    return pred, new, info, mask


def divergence_audit(selected, dense, ego, previous, meta, own_masks):
    """Prove same-input backend fidelity and measure prior guard differences."""
    a, sa, ia, ma = capture_step(selected, ego, previous['selected'], meta)
    b, sb, ib, mb = capture_step(dense, ego, previous['selected'], meta)
    errors = {k:float((a[k]-b[k]).abs().max()) for k in a}
    map_errors = [float((x-y).abs().max()) for x,y in zip(sa['spatial']['maps'], sb['spatial']['maps'])]
    result = dict(same_input_masks_equal=bool(torch.equal(ma,mb)),
        same_input_prediction_max_errors=errors, same_input_map_max_errors=map_errors,
        same_input_numerically_close=all(torch.allclose(a[k],b[k],atol=1e-3,rtol=1e-3) for k in a)
            and all(torch.allclose(x,y,atol=1e-3,rtol=1e-3) for x,y in zip(sa['spatial']['maps'],sb['spatial']['maps'])),
        own_mask_xor_cells=int((own_masks['selected'] ^ own_masks['masked_dense']).sum()),
        own_selected_cells=int(own_masks['selected'].sum()), own_dense_cells=int(own_masks['masked_dense'].sum()))
    left, right = previous['selected']['spatial'], previous['masked_dense']['spatial']
    result['prior_reference_max_difference'] = float((left['reference']-right['reference']).abs().max())
    result['prior_map_max_differences'] = [float((x-y).abs().max()) for x,y in zip(left['maps'],right['maps'])]
    threshold = selected.spatial.confidence_threshold
    if threshold is not None:
        x,y = left['confidence'],right['confidence']
        result['prior_confidence_max_difference'] = float((x-y).abs().max())
        flips = (x >= threshold) != (y >= threshold)
        result['prior_confidence_threshold_crossings'] = int(flips.sum())
        result['crossing_confidence_pairs'] = list(zip(x[flips].cpu().tolist()[:16],y[flips].cpu().tolist()[:16]))
        result['confidence_min_threshold_margin'] = min(float((x-threshold).abs().min()),float((y-threshold).abs().min()))
    return result

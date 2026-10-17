"""Exact wire-value forward with a selected-feature straight-through backward."""
import hashlib
import numpy as np
import torch
from opencood.utils.quantized_feature_packet import prepare


class WireValueSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selected, wire):
        return wire.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def budget_features(indices, features, activation, options):
    """Same selection/rounding as wire codec; no gradient through ranking/scales.

    Keep original sparse row order and zero untransmitted rows. Only selected
    code rows receive a surrogate gradient; ego features are never transformed.
    """
    address = indices.detach().cpu().numpy()
    values = features.detach().cpu().numpy()
    result = features * 0
    for batch in np.unique(address[:, 0]):
        rows = np.flatnonzero(address[:, 0] == batch)
        coords = address[rows, 1:]
        selected, codes, scales = prepare(coords, values[rows], activation, options)
        # uint16 contract from prepare makes this collision-free for 3D indices.
        keys = (coords[:, 0].astype(np.int64) << 32) + (coords[:, 1].astype(np.int64) << 16) + coords[:, 2]
        target = (selected[:, 0].astype(np.int64) << 32) + (selected[:, 1].astype(np.int64) << 16) + selected[:, 2]
        order = np.argsort(keys)
        chosen = rows[order[np.searchsorted(keys[order], target)]]
        chosen = torch.as_tensor(chosen, device=features.device, dtype=torch.long)
        quantized = torch.as_tensor(codes.astype(np.float32) * scales, device=features.device, dtype=features.dtype)
        result = result.index_copy(0, chosen, WireValueSTE.apply(features.index_select(0, chosen), quantized))
    return result


def decoder_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(('backbone_2d.', 'cls_head.', 'reg_head.', 'dir_head.')):
            digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

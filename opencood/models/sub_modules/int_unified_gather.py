"""Inference-only selected-cell execution of the existing V2 update weights.

Two spatial convolutions become two gathered matrix multiplications. Dependencies
are expanded by one cell per layer; outside-image activations are zero *after*
normalization and bias, matching Conv2d padding at every layer.
"""
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_unified_bev import ResidualStateUpdate


def _pointwise(x, conv):
    return F.linear(x, conv.weight[:, :, 0, 0], conv.bias)


def _indices(mask):
    return mask.reshape(-1).nonzero(as_tuple=False).flatten()


def _lookup(indices, size):
    # The final row is a zero sentinel for out-of-image neighbors.
    table = indices.new_full((size,), indices.numel())
    table[indices] = torch.arange(indices.numel(), device=indices.device)
    return table


def _neighbors(target, table, height, width, sentinel):
    offsets = torch.arange(-1, 2, device=target.device)
    yy = target[:, None, None] // width + offsets[None, :, None]
    xx = target[:, None, None] % width + offsets[None, None, :]
    valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    flat = (yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)).reshape(-1)
    return torch.where(valid.reshape(-1), table[flat], sentinel).reshape(-1, 9)


def _residual(x, source, target, block, height, width):
    table = _lookup(source, height * width)
    skip = x[table[target]]
    net = block.net
    hidden = F.silu(_pointwise(net[0].norm(x), net[1]))
    hidden = torch.cat([hidden, hidden.new_zeros((1, hidden.shape[1]))])
    neighbors = _neighbors(target, table, height, width, source.numel())
    patches = hidden[neighbors].reshape(target.numel(), -1)
    conv = net[3]
    # Gather order is kernel-y, kernel-x, channel, not Conv2d's channel,y,x.
    weight = conv.weight.permute(0, 2, 3, 1).reshape(conv.out_channels, -1)
    mixed = F.silu(F.linear(patches, weight, conv.bias))
    return skip + _pointwise(mixed, net[5])


def gathered_update(module, packed, mask):
    """Return old state except selected cells, without changing model parameters.

    This experimental executor is deliberately not wired into production routing.
    Mask is an exact cell mask (tile expansion, if any, belongs to the caller).
    Index construction, tensor layout conversions and scatter are part of its cost.
    """
    if not isinstance(module, ResidualStateUpdate):
        raise TypeError('Only the V2 ResidualStateUpdate is supported')
    if module.training or torch.is_grad_enabled():
        raise ValueError('Gathered update is frozen inference only')
    if packed.ndim != 4 or packed.shape[0] != 1:
        raise ValueError('Expected a single packed BEV frame')
    height, width = packed.shape[-2:]
    if mask.shape != (1, 1, height, width) or mask.dtype != torch.bool:
        raise ValueError('Expected a boolean cell mask matching the BEV grid')
    for block in [module.net[3], module.net[4]]:
        conv = block.net[3]
        if conv.kernel_size != (3, 3) or conv.dilation != (1, 1) or conv.groups != 1:
            raise ValueError('Unsupported spatial update configuration')
    old = packed[:, module.evidence:module.evidence + module.channels]
    selected = _indices(mask)
    if not selected.numel():
        return old.clone()
    one = _indices(F.max_pool2d(mask.float(), 3, stride=1, padding=1).bool())
    two = _indices(F.max_pool2d(mask.float(), 5, stride=1, padding=2).bool())
    rows = packed[0].permute(1, 2, 0).reshape(height * width, -1)
    net = module.net
    value = F.silu(net[1].norm(_pointwise(rows[two], net[0])))
    value = _residual(value, two, one, net[3], height, width)
    value = _residual(value, one, selected, net[4], height, width)
    residual, gate = _pointwise(net[5].norm(value), net[6]).chunk(2, dim=1)
    current = rows[selected]
    retain = gate.sigmoid() * current[:, -1:]
    value = retain * current[:, module.evidence:module.evidence + module.channels]
    value = value + (1 - retain) * (_pointwise(current[:, :module.evidence], module.current) + residual)
    result = old[0].permute(1, 2, 0).reshape(height * width, module.channels).clone()
    result[selected] = value
    return result.reshape(height, width, module.channels).permute(2, 0, 1).unsqueeze(0).contiguous()

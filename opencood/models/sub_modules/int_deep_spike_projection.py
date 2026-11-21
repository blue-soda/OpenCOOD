"""Frozen, signed event updates to the linear (pre-BN) deep projection."""
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection


class DeepSpikeProjection(DeepPhaseProjection):
    """The cache packs integer send levels and linear sums at deep grid sites."""
    accepts_quantized_readout = True

    def __init__(self, block, quantum, mode):
        super().__init__(block, indexed=True)
        if mode not in ('dense', 'amplitude', 'unit'):
            raise ValueError('Unknown projection mode')
        self.mode = mode
        conv = block[0]
        self.quantum = torch.as_tensor(quantum, device=conv.weight.device,
                                       dtype=torch.float32).reshape(1, -1, 1, 1)
        if self.quantum.numel() != conv.in_channels or not bool(
                (torch.isfinite(self.quantum) & (self.quantum > 0)).all()):
            raise ValueError('Expected a positive finite quantum per input channel')
        w = conv.weight.detach().double() * self.quantum.reshape(-1, 1, 1, 1)
        maximum = w.abs().amax(dim=(0, 2, 3), keepdim=True)
        scale = torch.where(maximum > 0, maximum / 32767, torch.ones_like(maximum))
        self.scale = scale.reshape(1, -1, 1, 1).float()
        self.weight = torch.round(w / scale).short().flatten(1).contiguous()
        self.weight_bound = int(self.weight.long().abs().sum(dim=0).max())
        self.ledger = {}

    def levels(self, deep):
        values = (deep / self.quantum).round()
        if not bool(torch.isfinite(values).all()) or float(values.abs().max()) >= 2**31:
            raise ValueError('Nonfinite or out-of-range send level')
        levels = values.to(torch.int64)
        if int(levels.abs().max()) * self.weight_bound >= 2**51:
            raise ValueError('Input exceeds exact dense and incremental sum range')
        return levels

    def dense_linear(self, levels):
        # A double GEMM is an exact independent integer-sum reference in this range.
        size = levels.shape
        linear = self.weight.double().T @ levels.flatten(2).double()
        return linear.long().reshape(size[0], -1, *size[-2:])

    def event_linear(self, changes, linear, unit_weight=None):
        flat = changes.flatten(2)[0]
        channels, sites = flat.nonzero(as_tuple=True)
        values = flat[channels, sites]
        fanout = self.weight.shape[1]
        # Bound temporary storage, including when a single event has high amplitude.
        chunk = max(1, 1048576 // fanout)
        out = linear.flatten(2)[0]
        if self.mode == 'unit':
            table = self.weight if unit_weight is None else unit_weight
            for sign in (-1, 1):
                selected = values.sign() == sign
                ends = values[selected].abs().cumsum(0)
                if not ends.numel():
                    continue
                src, dst = channels[selected], sites[selected]
                units = int(ends[-1])
                for start in range(0, units, chunk):
                    positions = torch.arange(start, min(start + chunk, units), device=flat.device)
                    idx = torch.searchsorted(ends, positions, right=True)
                    weights = table[src[idx]].long()
                    out.index_add_(1, dst[idx], (weights if sign > 0 else -weights).T)
        else:
            for start in range(0, len(values), chunk):
                slot = slice(start, start + chunk)
                weighted = self.weight[channels[slot]].long() * values[slot, None]
                out.index_add_(1, sites[slot], weighted.T)

    def finish(self, levels, linear, changes, rebuilt, decode_scale=None):
        conv, bn, relu = self.block
        scale = self.scale if decode_scale is None else decode_scale
        image = F.pixel_shuffle(linear.float(), self.stride) * scale
        if conv.bias is not None:
            image = image + conv.bias.view(1, -1, 1, 1)
        nonzero = int(torch.count_nonzero(changes))
        units = int(changes.abs().sum())
        fanout = self.weight.shape[1]
        self.ledger = dict(rebuilt=rebuilt, nonzero=nonzero, units=units,
                           dense_mac=levels.numel()*fanout,
                           amplitude_mac=nonzero*fanout, unit_ac=units*fanout)
        return relu(bn(image)), torch.cat((levels, linear), dim=1)

    def fresh(self, deep):
        levels = self.levels(deep)
        return self.finish(levels, self.dense_linear(levels), levels, True)

    def advance(self, deep, written, previous, past_pose, current_pose, bounds, route=None):
        if route is None:
            raise ValueError('Shared nearest-neighbor route required')
        indices, missing = route
        aligned = previous.flatten(2).index_select(2, indices).view_as(previous)
        aligned.masked_fill_(missing, 0)
        levels = self.levels(deep)
        before, linear = aligned.split((levels.shape[1], self.weight.shape[1]), dim=1)
        changes = levels - before
        if bool((changes.masked_select(~(written | missing)) != 0).any()):
            raise ValueError('Unexpected change outside write support')
        if self.mode == 'dense':
            linear = self.dense_linear(levels)
        else:
            self.event_linear(changes, linear)
        return self.finish(levels, linear, changes, False)

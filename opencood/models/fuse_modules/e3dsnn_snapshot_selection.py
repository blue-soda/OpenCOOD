"""Hard byte-limited snapshot selection with the existing training gradients.

The codec decides the actual forward content. Only IF fixed-frontier STE or
the original ANN quantizer/row mask supplies gradients; sorting and bytes
remain discrete. Geometry/basis validation is performed by the wire codec.
"""
from dataclasses import dataclass
import numpy as np
import torch
from opencood.utils import task_snapshot_packet as wire
from opencood.models.fuse_modules.e3dsnn_content_message import CountContent


@dataclass
class SelectedSnapshot:
    packet: bytes
    counts: torch.Tensor
    scales: torch.Tensor
    record: dict

    def ann_events(self):
        message = CountContent(self.counts, self.scales)
        return [message.event(k) for k in (1, 2, 3)]


def select_training_snapshot(full, scales, metadata, budget, stream=None,
                             row_order=None):
    """Select once, retaining P gradients without differentiating wire bytes.

    `full` and `stream` must refer to the same sender frame/forward. ANN rows
    contain their original complete codes; even zero-code channels in accepted
    rows retain the original quantizer surrogate. Silence carries unit scales.
    """
    family = metadata['family']
    if scales.shape != (1, 32, 1, 1, 1):
        raise ValueError('Expected original per-channel scales')
    if family == 'if8':
        if stream is None or row_order is not None:
            raise ValueError('IF selection requires its original event stream')
        order = dict(addresses=stream.addresses)
    elif family == 'ann2bit':
        if stream is not None or row_order is None:
            raise ValueError('ANN selection requires its original row ordering')
        order = dict(row_order=row_order)
    else:
        raise ValueError('Unknown snapshot family')
    packet, selected, record = wire.select(
        full.detach().cpu().numpy(), scales.detach().cpu().numpy().reshape(32),
        metadata, budget, **order)
    hard = full.new_tensor(selected)
    if family == 'if8':
        n = len(stream.addresses) if budget is None else record['accepted_units']
        counts = stream.prefix(n)
    else:
        accepted_rows = (hard != 0).any(1, keepdim=True).to(full.dtype)
        counts = full * accepted_rows
    if not torch.equal(counts.detach(), hard):
        raise ValueError('Training surrogate changed actual wire content')
    restored, restored_scales = wire.decode(packet, metadata)
    if not np.array_equal(restored, selected):
        raise ValueError('Snapshot reconstruction changed counts')
    paid_scales = full.new_tensor(restored_scales).reshape(1, 32, 1, 1, 1)
    if packet and not torch.equal(paid_scales, scales.detach()):
        raise ValueError('Original scales differ from paid scales')
    return SelectedSnapshot(packet, counts, paid_scales, record)

"""Hard byte-limited snapshot selection with the existing training gradients.

The codec decides the actual forward content. Only IF fixed-frontier STE or
the original ANN quantizer/row mask supplies gradients; sorting and bytes
remain discrete. Geometry/basis validation is performed by the wire codec.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from opencood.utils import task_snapshot_packet as wire
from opencood.models.fuse_modules.e3dsnn_content_message import CountContent
from opencood.models.fuse_modules.e3dsnn_event_stream import _HardCounts


def select_event_quota(stream, quota):
    """Tensor-only IF prefix selection; quota is events, never measured bytes."""
    if quota is not None and (type(quota) is not int or quota < 0):
        raise ValueError('Expected a nonnegative event quota or None')
    count = len(stream.addresses) if quota is None else min(quota, len(stream.addresses))
    counts = stream.prefix(count)
    scales = counts.new_ones((1,32,1,1,1))
    return SelectedSnapshot(None, counts, scales,
        dict(accepted_units=count,selected_events=count,event_quota=quota,
             selection='IF crossing-time prefix, fixed event quota',communication_measured=False))


def select_if_window(drive, window, pilot=None):
    """Counts accumulated by a fixed time in a unit-horizon, constant-drive IF8 stream."""
    if not 0 <= window <= 1:
        raise ValueError('Expected an IF window between zero and one')
    bounded = drive.clamp(0, 8).double()
    if pilot is not None:
        # Only crossings that are already visible at the pilot time may influence the deadline.
        early_count = (bounded * pilot['time']).floor().sum()
        factor = ((early_count.log1p() - pilot['mean_log_count']) * pilot['slope']).exp()
        window = (window * factor).clamp(min=pilot['time'], max=1).item()
    counts = (bounded * window).floor().to(drive.dtype)
    events = int(counts.sum().item())
    return SelectedSnapshot(None, counts, counts.new_ones((1,32,1,1,1)),
        dict(selected_events=events, accepted_units=events, if_window=window,
             selection='IF crossing-time prefix, fixed time window', communication_measured=False))


def select_tensor_quota(drive, quota):
    """Select the IF8 time prefix on-device, breaking equal-time ties by address."""
    if quota is not None and (type(quota) is not int or quota < 0):
        raise ValueError('Expected a nonnegative event quota or None')
    if (drive.ndim != 5 or drive.shape[0] != 1 or not drive.numel() or
            drive.numel() >= 2**32 or not torch.isfinite(drive).all() or (drive < 0).any()):
        raise ValueError('Expected finite nonnegative single-frame drive')
    bounded = drive.clamp(max=8)
    flat = bounded.detach().flatten()
    # nonzero returns addresses in increasing order, including on CUDA.
    active = torch.nonzero(flat >= 1, as_tuple=True)[0]
    rates = flat[active].double()
    total = int(rates.floor().sum().item())
    count = total if quota is None else min(quota, total)
    if count == 0:
        counts = bounded * 0
    elif count == total:
        counts = _HardCounts.apply(bounded, bounded.detach().floor())
    else:
        levels = torch.arange(1, 9, device=drive.device, dtype=torch.float64)[:, None]
        times = levels / rates
        frontier = torch.kthvalue(times[times <= 1], count).values
        earlier = (times < frontier).sum(dim=0)
        tied = (times == frontier).any(dim=0)
        admitted = tied & (tied.cumsum(dim=0) <= count - earlier.sum())
        sparse_counts = (earlier + admitted).to(drive.dtype)
        hard = torch.zeros_like(flat).index_copy_(0, active, sparse_counts)
        counts = _HardCounts.apply(bounded * frontier.to(drive.dtype), hard.reshape_as(drive))
    return SelectedSnapshot(None, counts, counts.new_ones((1,32,1,1,1)),
        dict(accepted_units=count, selected_events=count, event_quota=quota,
             selection='IF crossing-time prefix, fixed event quota', communication_measured=False))


@dataclass
class SelectedSnapshot:
    packet: Optional[bytes]
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

"""Causal thresholded messages for a paired shallow-state intervention."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_codebook import pack_symbols, unpack_symbols
from opencood.models.sub_modules.int_fine_demand import fine_support


@dataclass
class Candidates:
    levels: torch.Tensor
    ids: torch.Tensor
    change: torch.Tensor


class SilentReceiver:
    def __init__(self, threshold, floating=False, forget=False):
        self.threshold = threshold
        self.floating = floating
        self.forget = forget
        self.offset = None
        self.levels = None
        self.capture_sites = False

    def refresh(self, value):
        self.levels = value.clone() if self.floating else (value * 4).round().int()
        self.offset = torch.zeros_like(self.levels) if self.forget else None

    def prepare(self, value, sites, warp):
        aligned = warp(self.levels).contiguous()
        receiver = aligned.flatten(2)[0]
        raw = value.flatten(2)[0, :, sites]
        target = raw if self.floating else (raw * 4).round().int()
        delta = target - receiver[:, sites]
        threshold = self.threshold / 4 if self.floating else self.threshold
        if self.forget:
            self.offset = warp(self.offset).contiguous()
            offset = self.offset.flatten(2)[0]
            delta = delta - offset[:, sites]
            # Discard only the innovations below threshold, without changing
            # the physical receiver, budget-overflow events or codebook error.
            offset[:, sites] += torch.where(delta.abs() < threshold, delta, 0)
        channel, slot = (delta.abs() >= threshold).nonzero(as_tuple=True)
        return Candidates(aligned, channel * receiver.shape[1] + sites[slot], delta[channel, slot])

    def receive(self, candidates, limit, frame, read, book):
        ids, change = candidates.ids, candidates.change
        demand = ids.numel()
        if demand > limit:
            magnitude = change.abs().contiguous()
            if self.floating:
                magnitude = magnitude.view(torch.int32)
            tie = ((ids + frame) * 2654435761) % (1 << 32)
            chosen = torch.topk(magnitude.long() * (1 << 32) + tie, limit, sorted=False).indices
            ids, change = ids[chosen], change[chosen]
        sent = change
        payload_bytes = 4 * len(ids)
        if not self.floating:
            table = change.new_tensor(book)
            rank = torch.bucketize(2 * change.abs(), table[1:] + table[:-1])
            payload = pack_symbols(rank + 4 * (change < 0).long())
            codes = unpack_symbols(payload, len(ids))
            sent = table[codes % 4] * torch.where(codes >= 4, -1, 1).int()
            payload_bytes = payload.numel()
        self.levels = candidates.levels
        self.levels.view(-1).index_add_(0, ids, sent)
        active = torch.zeros_like(read)
        active.view(-1)[ids % read.numel()] = True
        if self.capture_sites:
            self.event_sites = active
        read_tiles = F.max_pool2d(read.float(), 8, 8, ceil_mode=True).bool()
        sent_tiles = F.max_pool2d(active.float(), 8, 8, ceil_mode=True).bool()
        stats = dict(eligible=demand, addresses=len(ids), bytes=8 * len(ids) + payload_bytes,
                     read_tiles=int(read_tiles.sum()), silent_tiles=int((read_tiles & ~sent_tiles).sum()))
        if not self.floating:
            stats['units'] = int(sent.abs().sum(dtype=torch.int64))
        return self.levels if self.floating else self.levels.float() * .25, stats


class PairedEdge:
    """Compute the current joint grant without future inputs, labels or detections."""
    def __init__(self, bounds, cap, book, thresholds, calibration, native=False, factorial=False):
        self.bounds, self.cap, self.book = bounds, cap, book
        self.calibration = calibration
        self.native = native
        if calibration:
            self.arms = {str(t): SilentReceiver(t) for t in thresholds}
        else:
            t, = thresholds
            self.arms = dict(carry=SilentReceiver(t), reset=SilentReceiver(t, forget=True),
                             float_address=SilentReceiver(t, floating=True), float_bytes=SilentReceiver(t, floating=True))
            if native:
                del self.arms['float_bytes']
            if factorial:
                self.arms['float_address_reset'] = SilentReceiver(t, floating=True, forget=True)
                self.arms['float_bytes_reset'] = SilentReceiver(t, floating=True, forget=True)

    def start(self, meta, shape, reason):
        self.meta, self.shape, self.reason = meta, shape, reason
        self.result = {}
        self.stats = {}

    def apply(self, name, value, active=None):
        if self.result:
            assert torch.equal(value, self.current) and (self.reason or torch.equal(active, self.active))
            return self.result[name]
        self.current, self.active = value, active
        if self.reason:
            for key, arm in self.arms.items():
                arm.refresh(value)
                self.result[key] = value
                self.stats[key] = dict(bytes=value.numel() * value.element_size())
        else:
            read = fine_support(active, value.shape[-2:])
            sites = read.flatten().nonzero().flatten()
            area = block_lidar_range(self.bounds, self.shape, value.shape[-2:], 2)

            def warp(x):
                moved = warp_bev(x.float(), self.pose, self.meta['pose'], area)[0]
                return moved.to(x.dtype)

            candidates = {key: arm.prepare(value, sites, warp) for key, arm in self.arms.items()}
            limits = {key: min(self.cap, len(c.ids)) for key, c in candidates.items()}
            if not self.calibration and not self.native:
                # A floating record costs 12 bytes. A four-level record uses
                # 8 address bytes and three packed amplitude bits.
                common = min(96 * limit // 67 if key.startswith('float_bytes') else limit
                             for key, limit in limits.items())
                byte_grant = (8 * common + (3 * common + 7) // 8) // 12
                limits = {key: byte_grant if key.startswith('float_bytes') else common for key in candidates}
            for key, arm in self.arms.items():
                self.result[key], self.stats[key] = arm.receive(
                    candidates[key], limits[key], int(self.meta['frame']), read, self.book)
        self.pose = self.meta['pose']
        return self.result[name]


class EdgeView:
    def __init__(self, paired, name):
        self.paired, self.name = paired, name
        self.active = None

    @property
    def phase(self):
        return 'refresh' if self.paired.reason else 'ordinary'

    def __call__(self, value):
        return self.paired.apply(self.name, value, self.active)

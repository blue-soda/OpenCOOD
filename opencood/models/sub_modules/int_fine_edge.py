"""Quantized fine-to-middle messages with a causal, ego-aligned receiver."""
import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev


class FineEdgeSender:
    def __init__(self, bounds, dense=False):
        self.bounds = bounds
        self.dense = dense
        self.quantum = .25

    def begin(self, previous, meta, spatial_shape, reason):
        self.previous = None if reason else previous
        self.pose = meta['pose']
        self.shape = spatial_shape
        self.levels = None

    def __call__(self, value):
        if self.levels is not None:
            raise RuntimeError('Duplicate fine encoding on one scan')
        target = (value/self.quantum).round()
        if not bool(target.abs().max() < 2**23):
            raise ValueError('Fine send level outside exact FP32 warp range')
        target = target.int()
        self.ledger = dict(cells=target.numel(), cache_bytes=target.numel()*target.element_size(),
                           nonzero=int(target.count_nonzero()))
        if self.previous is None:
            self.levels = target
            self.ledger['refresh_levels'] = target.numel()
        else:
            area = block_lidar_range(self.bounds,self.shape,value.shape[-2:],2)
            old = warp_bev(self.previous['fine_levels'].float(), self.previous['pose'], self.pose, area)[0].int()
            delta = target-old
            indices = delta.flatten().nonzero().flatten()
            changes = delta.flatten()[indices]
            self.ledger.update(addresses=indices.numel(), units=int(changes.abs().sum(dtype=torch.int64)))
            if self.dense:
                self.levels = target
            else:
                self.levels = old.contiguous()
                self.levels.view(-1).index_add_(0,indices,changes)
                assert torch.equal(self.levels,target)
        self.previous = None
        return self.levels.float()*self.quantum

    def commit(self, state, drop):
        if self.levels is None:
            raise RuntimeError('Missing fine-to-middle receiver call')
        state['fine_levels'] = self.levels
        residual = state['value']-self.levels.float()*self.quantum
        self.ledger['residual_l1'] = float(residual.abs().sum(dtype=torch.float64))
        assert float(residual.abs().max()) <= self.quantum/2
        if drop:
            state['value'] = self.levels.float()*self.quantum
        self.levels = None


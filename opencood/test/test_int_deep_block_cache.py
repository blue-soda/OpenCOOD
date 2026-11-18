import unittest

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from test_int_pyramid_cache import fixture


class DeepBlockCacheTests(unittest.TestCase):
    def setUp(self):
        self.source = ProtectedMultiscaleRefreshDetector(fixture())
        self.cache = DeepBlockCache(self.source, tile=2)
        self.spatial = torch.randn(1, 64, 80, 120)
        self.meta = dict(timestamp_us=0, segment='a', pose=np.eye(4))

    @torch.no_grad()
    def test_empty_optional_mask_skips_heavy_and_retains_deep_state(self):
        _, old, _ = self.cache.step_spatial(self.spatial, None, self.meta)
        next_meta = dict(self.meta, timestamp_us=100000)
        calls = []
        handle = self.cache.heavy.encoder.register_forward_hook(lambda *args: calls.append(True))
        try:
            _, new, info = self.cache.step_spatial(self.spatial, old, next_meta)
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(new['deep'], old['deep']))
        self.assertTrue(torch.allclose(new['deep_age'], torch.full_like(old['deep_age'], .1)))
        self.assertEqual(info['actual_write_fraction'], 0.)

    @torch.no_grad()
    def test_local_and_dense_reference_share_persistent_writeback(self):
        _, old, _ = self.cache.step_spatial(self.spatial, None, self.meta)
        next_meta = dict(self.meta, timestamp_us=100000)
        mask = torch.zeros_like(old['deep_age'], dtype=torch.bool)
        mask[..., :2, :2] = True
        pred, local, _ = self.cache.step_spatial(self.spatial, old, next_meta, mask)
        ref, dense, _ = self.cache.step_spatial(self.spatial, old, next_meta, mask, dense_reference=True)
        for key in (*self.source.state_keys, 'deep_age'):
            self.assertTrue(torch.allclose(local[key], dense[key], atol=1e-4, rtol=1e-4), key)
        for key in pred:
            self.assertTrue(torch.allclose(pred[key], ref[key], atol=1e-4, rtol=1e-4), key)
        self.assertTrue(torch.equal(local['deep_age'][mask], old['deep_age'][mask]))
        self.assertTrue(torch.all(local['deep_age'][~mask] > 0))

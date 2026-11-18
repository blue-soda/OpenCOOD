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
        handle = self.cache.stage.encoder.register_forward_hook(lambda *args: calls.append(True))
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

    @torch.no_grad()
    def test_light_all_write_matches_p4_with_variable_time_steps(self):
        cache = DeepBlockCache(self.source, tile=2, operator='light')
        local = baseline = None
        for stamp in (0, 92000, 217000, 300000, 437000):
            meta = dict(self.meta, timestamp_us=stamp)
            mask = None if local is None else torch.ones_like(local['deep_age'], dtype=torch.bool)
            pred, local, _ = cache.step_spatial(self.spatial, local, meta, mask)
            expected, baseline, _ = self.source.step_spatial(self.spatial, baseline, meta)
            for key in self.source.state_keys:
                self.assertTrue(torch.equal(local[key], baseline[key]), key)
            for key in pred:
                self.assertTrue(torch.equal(pred[key], expected[key]), key)

    @torch.no_grad()
    def test_light_crop_matches_same_mask_on_valid_boundary(self):
        cache = DeepBlockCache(self.source, tile=2, operator='light')
        _, old, _ = cache.step_spatial(self.spatial, None, self.meta)
        pose = self.meta['pose'].copy()
        pose[0, 3] = 3.
        meta = dict(self.meta, timestamp_us=120000, pose=pose)
        mask = torch.zeros_like(old['deep_age'], dtype=torch.bool)
        mask[..., 4:6, 4:6] = True
        pred, local, info = cache.step_spatial(self.spatial, old, meta, mask)
        ref, dense, _ = cache.step_spatial(self.spatial, old, meta, mask, dense_reference=True)
        self.assertLess(info['stage_conv_ratio'], 1.)
        for key in (*self.source.state_keys, 'deep_age'):
            self.assertTrue(torch.allclose(local[key], dense[key], atol=1e-4, rtol=1e-4), key)
        for key in pred:
            self.assertTrue(torch.allclose(pred[key], ref[key], atol=1e-4, rtol=1e-4), key)

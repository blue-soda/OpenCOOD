import unittest

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_pyramid_regions import (
    plan_coalesced_refresh, plan_counted_host_refresh, plan_host_coalesced_refresh,
)
from test_int_pyramid_cache import fixture


class DeepPhaseProjectionTests(unittest.TestCase):
    @torch.no_grad()
    def test_stride_four_reuses_aligned_sites_and_projects_new_writes(self):
        torch.manual_seed(17)
        block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(5, 7, 4, stride=4, bias=False),
            torch.nn.BatchNorm2d(7),
            torch.nn.ReLU(),
        ).eval()
        block[1].bias.copy_(torch.linspace(-.2, .2, 7))
        projector = DeepPhaseProjection(block)
        previous = torch.randn(1, 5, 3, 6)
        old_output, old_phases = projector.fresh(previous)
        old_pose = np.eye(4)
        moved = old_pose.copy()
        moved[0, 3] = 1.
        bounds = (0, 0, -1, 6, 3, 1)
        aligned, valid = warp_bev(previous, old_pose, moved, bounds)
        written = ~valid
        written[..., 1, 2] = True
        deep = torch.where(written, torch.randn_like(aligned), aligned)
        sizes = []
        hook = block[0].register_forward_pre_hook(lambda _, inputs: sizes.append(inputs[0].shape[-2:]))
        try:
            projected, _ = projector.advance(deep, written, old_phases, old_pose, moved, bounds)
            held, _ = projector.advance(previous, torch.zeros_like(written), old_phases, old_pose, old_pose, bounds)
            refreshed, _ = projector.advance(deep, torch.ones_like(written), old_phases, old_pose, moved, bounds)
        finally:
            hook.remove()
        self.assertEqual(sizes, [(1, 4), deep.shape[-2:]])
        self.assertTrue(torch.allclose(projected, block(deep), atol=1e-6))
        self.assertTrue(torch.equal(held, old_output))
        self.assertTrue(torch.equal(refreshed, block(deep)))


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

    def test_host_planning_matches_partial_tiles_and_dense_fallback(self):
        shape = (50, 126)
        request = torch.zeros(1, 1, 25, 63, dtype=torch.bool)
        for cells in ((), ((0, 1),), ((7, 61), (24, 62), (12, 28))):
            request.zero_()
            for y, x in cells:
                request[..., y, x] = True
            with self.subTest(cells=cells):
                ref = plan_coalesced_refresh(request, shape, 5, 8)
                host = plan_host_coalesced_refresh(request, shape, 5, 8)
                self.assertTrue(torch.equal(ref[0], host[0]))
                self.assertEqual(ref[1:], host[1:])
        request.fill_(True)
        ref = plan_coalesced_refresh(request, shape, 5, 8)
        host = plan_host_coalesced_refresh(request, shape, 5, 8)
        self.assertTrue(torch.equal(ref[0], host[0]))
        self.assertEqual(ref[1:], host[1:])
        self.assertIsNone(host[1])

    @torch.no_grad()
    def test_direct_region_batch_matches_packed_on_odd_boundaries(self):
        stage = DeepBlockCache(self.source, tile=2, operator='light').stage
        current = torch.randn(1, stage.input_channels, 13, 19)
        old = torch.randn(1, stage.output_channels, 7, 10)
        valid = torch.rand(1, 1, 7, 10) > .3
        packed = stage.pack(current, old, valid)
        jobs = [(0,0,0,0,0,0,7,9), (0,0,0,0,6,10,13,19)]
        crops = torch.cat([packed[..., j[4]:j[6], j[5]:j[7]] for j in jobs])
        self.assertTrue(torch.equal(stage(crops), stage(*stage.region_inputs(current, old, valid, jobs))))
        self.assertTrue(torch.equal(stage(packed), stage(*stage.region_inputs(current, old, valid, None))))

    def test_counted_planning_preserves_partial_edge_tile_fractions(self):
        request = torch.zeros(1, 1, 25, 63, dtype=torch.bool)
        shape = (50, 126)
        self.assertIsNone(plan_counted_host_refresh(request, shape, 5, 8))
        for indices in ([(0, 0), (24, 62), (23, 62)], [(24, 62)], None):
            request.fill_(indices is None)
            for y, x in indices or ():
                request[..., y, x] = True
            ref = plan_host_coalesced_refresh(request, shape, 5, 8)
            counted = plan_counted_host_refresh(request, shape, 5, 8)
            self.assertTrue(torch.equal(ref[0], counted.layout[0]))
            self.assertEqual(ref[1:], counted.layout[1:])
            self.assertAlmostEqual(float(request.float().mean()), counted.requested_fraction)
            self.assertAlmostEqual(float(ref[0].float().mean()), counted.written_fraction)

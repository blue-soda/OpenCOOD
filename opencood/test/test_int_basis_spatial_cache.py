import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class BasisCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.stage = SpatialBasisStage(nn.Sequential(*[BasicBlock(16, 16) for _ in range(3)]).eval(), 9).eval()
        self.bounds = [-16, -12, -2, 16, 12, 2]
        self.meta = dict(segment='a', timestamp_us=0, pose=np.eye(4))
        self.x = torch.rand(1, 16, 24, 32)

    def test_period1_equivalent(self):
        cache = SpatialBasisCache(self.stage, self.bounds, period=1)
        state = None
        with torch.no_grad():
            for frame in range(3):
                output, state, info = cache.step(self.x, state, dict(self.meta, timestamp_us=frame*100000))
                self.assertTrue(torch.equal(output, self.stage(self.x)))
                self.assertTrue(info['spatial_refresh'])

    def test_selected_matches_dense_own_recurrence(self):
        selected = SpatialBasisCache(self.stage, self.bounds, .5)
        dense = SpatialBasisCache(self.stage, self.bounds, .5, execution='masked_dense')
        ls = rs = None; refreshes = []
        with torch.no_grad():
            for frame in range(7):
                pose = np.eye(4); pose[0, 3] = frame
                meta = dict(self.meta, pose=pose, timestamp_us=frame*100000, segment='b' if frame >= 5 else 'a')
                x = self.x.clone(); x[:, :, 10:14, 14:18] += .3*frame
                a, ls, ia = selected.step(x, ls, meta); b, rs, ib = dense.step(x, rs, meta)
                self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=1e-5))
                self.assertEqual(ia['spatial_selected_cells'], ib['spatial_selected_cells'])
                refreshes.append(ia['spatial_refresh'])
                self.assertEqual(ls['stamp'], frame*100000)
        self.assertEqual(refreshes, [True, False, False, True, False, True, False])

    def test_empty_reuse_preserves_complete_maps_and_reference(self):
        cache = SpatialBasisCache(self.stage, self.bounds, threshold=1e9)
        with torch.no_grad():
            _, old, _ = cache.step(self.x, None, self.meta)
            snapshots = [x.clone() for x in old['maps']+[old['reference']]]
            _, new, info = cache.step(self.x*1.1, old, dict(self.meta, timestamp_us=100000))
        self.assertEqual(info['spatial_selected_cells'], 0)
        self.assertEqual(info['spatial_extra_linear_macs'], 0)
        self.assertEqual(info['spatial_basis_executed_fraction'], 0.)
        for expected, retained, original in zip(snapshots, new['maps']+[new['reference']], old['maps']+[old['reference']]):
            self.assertTrue(torch.equal(expected, retained)); self.assertTrue(torch.equal(expected, original))

    def test_uncovered_forces_refresh_and_age_and_causality(self):
        cache = SpatialBasisCache(self.stage, self.bounds, threshold=1e9)
        with torch.no_grad():
            _, old, _ = cache.step(self.x, None, self.meta)
            pose = np.eye(4); pose[0, 3] = 40
            _, _, info = cache.step(self.x, old, dict(self.meta, pose=pose, timestamp_us=100000))
            self.assertEqual(info['spatial_write_fraction'], 1.)
            _, _, info = cache.step(self.x, old, dict(self.meta, timestamp_us=300000))
            self.assertEqual(info['spatial_refresh_reason'], 'age')
            with self.assertRaises(ValueError): cache.step(self.x, old, self.meta)
            _, _, info = cache.step(self.x, old, self.meta, reset=True)
            self.assertEqual(info['spatial_refresh_reason'], 'reset')


if __name__ == '__main__': unittest.main()

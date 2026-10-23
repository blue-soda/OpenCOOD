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

    def test_past_confidence_protects_cells_without_current_lookahead(self):
        cache = SpatialBasisCache(self.stage, self.bounds, threshold=1e9, confidence_threshold=.1)
        with torch.no_grad():
            _, old, _ = cache.step(self.x, None, self.meta)
            old['confidence'] = torch.zeros_like(self.x[:, :1])
            old['confidence'][:, :, 12, 16] = .2
            confidence = old['confidence'].clone()
            _, new, info = cache.step(self.x, old, dict(self.meta, timestamp_us=100000))
        self.assertEqual(info['spatial_selected_cells'], 81)  # radius4 around one past confident cell
        self.assertAlmostEqual(info['spatial_guard_added_fraction'], 81./(24*32), places=6)
        self.assertTrue(torch.equal(old['confidence'], confidence))
        self.assertNotIn('confidence', new)  # detector must attach its OWN new prediction, never a future one
        with self.assertRaises(ValueError):
            SpatialBasisCache(self.stage, self.bounds, confidence_threshold=1.)

    def test_half_storage_keeps_compute_precision_and_static_values(self):
        full = SpatialBasisCache(self.stage,self.bounds,threshold=1e9)
        half = SpatialBasisCache(self.stage,self.bounds,threshold=1e9,cache_dtype=torch.float16)
        with torch.no_grad():
            a,sa,ia = full.step(self.x,None,self.meta)
            b,sb,ib = half.step(self.x,None,self.meta)
            self.assertTrue(torch.equal(a,b))
            expected_saving = sum(v.numel()*2 for v in sa['maps'])
            self.assertEqual(ia['spatial_state_bytes']-ib['spatial_state_bytes'],expected_saving)
            self.assertEqual(sb['reference'].dtype,torch.float32)
            original = [v.clone() for v in sb['maps']]
            c,sc,ic = half.step(self.x,sb,dict(self.meta,timestamp_us=100000))
            self.assertEqual(c.dtype,torch.float32)
            self.assertEqual(ic['spatial_selected_cells'],0)
            self.assertTrue(torch.equal(c,original[-1].float()))
            for old,saved,retained in zip(original,sb['maps'],sc['maps']):
                self.assertEqual(retained.dtype,torch.float16)
                self.assertTrue(torch.equal(old,saved))
                self.assertTrue(torch.equal(old,retained))

    def test_long_history_period_age_and_reset(self):
        cache = SpatialBasisCache(self.stage,self.bounds,threshold=1e9,period=5,max_full_age_s=.45)
        state = None; refresh = []
        with torch.no_grad():
            for frame in range(7):
                _,state,info = cache.step(self.x,state,dict(self.meta,timestamp_us=100000*frame))
                refresh.append(info['spatial_refresh'])
                self.assertLess(info['spatial_full_age'],.45)
            self.assertEqual(refresh,[True,False,False,False,False,True,False])
            _,initial,_ = cache.step(self.x,None,self.meta)
            _,_,info = cache.step(self.x,initial,dict(self.meta,timestamp_us=450000))
            self.assertEqual(info['spatial_refresh_reason'],'age')
            _,_,info = cache.step(self.x,initial,dict(self.meta,timestamp_us=100000),reset=True)
            self.assertEqual(info['spatial_refresh_reason'],'reset')
            with self.assertRaises(ValueError): cache.step(self.x,initial,self.meta)

    def test_half_storage_selected_matches_dense_mask_reference(self):
        selected = SpatialBasisCache(self.stage,self.bounds,.5,cache_dtype=torch.float16)
        dense = SpatialBasisCache(self.stage,self.bounds,.5,execution='masked_dense',cache_dtype=torch.float16)
        ls = rs = None
        with torch.no_grad():
            for frame in range(6):
                x = self.x.clone(); x[:,:,10:14,14:18] += frame*.7
                pose = np.eye(4); pose[0,3] = frame*.5
                meta = dict(self.meta,pose=pose,timestamp_us=frame*100000)
                a,ls,ia = selected.step(x,ls,meta); b,rs,ib = dense.step(x,rs,meta)
                self.assertEqual(a.dtype,torch.float32)
                self.assertTrue(torch.allclose(a,b,atol=2e-3,rtol=2e-3))
                self.assertEqual(ia['spatial_selected_cells'],ib['spatial_selected_cells'])
                self.assertTrue(all(v.dtype == torch.float16 and torch.isfinite(v).all() for v in ls['maps']))


if __name__ == '__main__': unittest.main()

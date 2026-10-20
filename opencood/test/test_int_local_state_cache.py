import unittest
import torch
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.models.sub_modules.int_local_state_cache import RegionalLocalStateDetector, change_request
from test_int_local_state import source_teacher


class RegionalStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(303)
        self.teacher = source_teacher()
        self.cfg = dict(model=dict(args=dict(lidar_range=[-24,-20,-1,24,20,1])))

    def meta(self, index, move=0.):
        pose = torch.eye(4).numpy(); pose[0, 3] = move
        return dict(timestamp_us=index*100000, segment='s', pose=pose)

    def model(self, **kwargs):
        return RegionalLocalStateDetector(self.teacher, self.cfg, **kwargs).eval()

    def test_full_refresh_preserves_recursive_dense_predictions(self):
        dense = LocalStateDetector(self.teacher, self.cfg, 'recurrent').eval()
        cached = self.model(fraction=1., tile=4)
        cached.load_state_dict(dense.state_dict(), strict=True)
        ds = cs = None
        with torch.no_grad():
            for i in range(4):
                x = torch.randn(1,64,40,48)
                expected, ds, _ = dense.step_spatial(x, ds, self.meta(i, i*.4))
                actual, cs, _ = cached.step_spatial(x, cs, self.meta(i, i*.4))
                self.assertTrue(all(torch.allclose(expected[k], actual[k], atol=1e-6) for k in expected))

    def test_no_change_reuses_without_calling_update_until_age_guard(self):
        cached = self.model(fraction=0., tile=4, max_age=.3)
        x = torch.randn(1,64,40,48)
        calls = []
        hook = cached.update.register_forward_pre_hook(lambda m, args: calls.append(args[0].shape))
        with torch.no_grad():
            _, first, _ = cached.step_spatial(x, None, self.meta(0))
            _, second, info = cached.step_spatial(x, first, self.meta(1))
            self.assertEqual(len(calls), 1)
            self.assertEqual(info['region_cache']['actual_write_fraction'], 0.)
            self.assertTrue(torch.equal(first['value'], second['value']))
            self.assertTrue(torch.equal(first['evidence_cache'], second['evidence_cache']))
            _, third, _ = cached.step_spatial(x, second, self.meta(2))
            _, fourth, info = cached.step_spatial(x, third, self.meta(3))
            self.assertEqual(len(calls), 2)
            self.assertEqual(float(fourth['cache_age'].max()), 0.)
            self.assertEqual(info['region_cache']['actual_write_fraction'], 1.)
        hook.remove()

    def test_sparse_matches_masked_dense_on_own_recursive_state_with_motion(self):
        sparse = self.model(fraction=.1, tile=4, fallback_fraction=1.)
        reference = self.model(fraction=.1, tile=4, fallback_fraction=1., execution='masked_dense')
        reference.load_state_dict(sparse.state_dict())
        ss = rs = None; saw_partial = False
        with torch.no_grad():
            for i in range(5):
                x = torch.randn(1,64,40,48)
                actual, ss, info = sparse.step_spatial(x, ss, self.meta(i, i*.4))
                expected, rs, _ = reference.step_spatial(x, rs, self.meta(i, i*.4))
                saw_partial |= 0 < info['region_cache']['actual_write_fraction'] < 1
                self.assertTrue(all(torch.allclose(actual[k], expected[k], atol=1e-5, rtol=1e-4) for k in actual))
                self.assertTrue(torch.allclose(ss['value'], rs['value'], atol=1e-5, rtol=1e-4))
                self.assertTrue(torch.equal(ss['cache_age'], rs['cache_age']))
        self.assertTrue(saw_partial)

    def test_change_ranking_and_new_coverage_are_not_whole_frame_guards(self):
        old = torch.ones(1,128,8,8); current = old.clone()
        current[..., :2, :2] += 3
        valid = torch.ones(1,1,8,8, dtype=torch.bool); valid[..., -1, -1] = False
        request = change_request(current, old, valid, torch.zeros_like(valid, dtype=torch.float),
                                 1/16, 2, .5)
        self.assertTrue(request[..., :2, :2].all())
        self.assertTrue(request[..., -1, -1].all())
        self.assertEqual(int(request.sum()), 5)

    def test_reset_clears_age_and_training_is_rejected(self):
        cached = self.model(fraction=0.)
        x = torch.randn(1,64,40,48)
        with torch.no_grad():
            _, state, _ = cached.step_spatial(x, None, self.meta(0))
            _, state, info = cached.step_spatial(x, state, self.meta(4))
            self.assertEqual(info['reset'], 'gap')
            self.assertEqual(float(state['cache_age'].max()), 0.)
        with self.assertRaises(ValueError):
            cached.train().step_spatial(x, None, self.meta(0))


if __name__ == '__main__':
    unittest.main()

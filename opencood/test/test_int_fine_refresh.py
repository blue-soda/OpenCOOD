import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector


class FineRefreshTests(unittest.TestCase):
    def test_period1_preserves_all_predictions_states_and_skips_new_functions(self):
        source = ProtectedMultiscaleRefreshDetector(fixture(), period=1)
        model = FineRefreshDetector(source)
        hits = []
        handles = [m.register_forward_hook(lambda *_: hits.append(1)) for m in (model.core.fine_update, model.core.fine_light)]
        a = b = None
        with torch.no_grad():
            for i in range(4):
                x = torch.randn(1, 64, 24, 32);pose = np.eye(4);pose[0, 3] = i * .15
                meta = dict(timestamp_us=i*100000, segment='a', pose=pose)
                pa, a, _ = source.step_spatial(x, a, meta);pb, b, _ = model.step_spatial(x, b, meta)
                self.assertTrue(all(torch.equal(pa[k], pb[k]) for k in pa))
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in source.state_keys))
        for h in handles:h.remove()
        self.assertEqual(hits, [])

    def test_three_heavy_encoders_skip_and_reset_matches_fresh(self):
        model = FineRefreshDetector(ProtectedMultiscaleRefreshDetector(fixture()))
        hits = [0, 0, 0]
        def hit(i):
            def count(*_):hits[i] += 1
            return count
        handles = [m.register_forward_hook(hit(i)) for i, m in enumerate((model.core.observation, model.core.coarse_encoder, model.core.deep_encoder))]
        state = None;x = torch.randn(1, 64, 24, 32)
        with torch.no_grad():
            for i in range(5):
                _, state, info = model.step_spatial(x, state, dict(timestamp_us=i*100000, segment='a', pose=np.eye(4)))
                self.assertEqual(hits, [1+int(i==4)]*3)
                self.assertEqual(info['cache_bytes'], sum(state[k].numel()*4 for k in model.state_keys))
            for extra in (dict(timestamp_us=700001, segment='a'), dict(timestamp_us=500000, segment='b'), dict(timestamp_us=500000, segment='a', reset_before=True)):
                meta = dict(extra, pose=np.eye(4));pred, new, _ = model.step_spatial(x, state, meta);fresh, fs, _ = model.step_spatial(x, None, meta)
                self.assertTrue(all(torch.equal(pred[k], fresh[k]) for k in pred))
                self.assertTrue(all(torch.equal(new[k], fs[k]) for k in model.state_keys))
            with self.assertRaises(ValueError):model.step_spatial(x, state, dict(timestamp_us=400000, segment='a', pose=np.eye(4)))
        for h in handles:h.remove()

    def test_own_state_gradients_reach_fine_light_without_changing_refresh(self):
        model = FineRefreshDetector(ProtectedMultiscaleRefreshDetector(fixture()))
        for name, p in model.named_parameters():p.requires_grad_(name.startswith(('core.fine_light.', 'core.fine_update.')))
        state = None
        for i in range(4):
            pred, state, _ = model.step_spatial(torch.randn(1, 64, 24, 32), state,
                dict(timestamp_us=i*100000, segment='a', pose=np.eye(4)), detach_state=False)
        pred['psm'].square().mean().backward()
        self.assertIsNotNone(state['value'].grad_fn)
        for module in (model.core.fine_light, model.core.fine_update):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.core.refresh_updates.parameters()))


if __name__ == '__main__':unittest.main()

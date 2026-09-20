import unittest

import torch

from opencood.models.sub_modules.ectra_geometry import warp_metric_bev
from opencood.models.sub_modules.ectra_recurrent_alignment import EctraRecurrentAlignment
from opencood.data_utils.ectra_ego_history import CausalEgoHistoryIndex


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.bounds = [-4.5, -4.5, -1, 4.5, 4.5, 1]
        self.x = torch.zeros(1, 1, 9, 9)
        self.x[0, 0, 4, 5] = 1
        self.t = torch.eye(4).unsqueeze(0)

    def test_identity_and_grad(self):
        x = self.x.clone().requires_grad_()
        y, mask = warp_metric_bev(x, self.t, self.bounds)
        torch.testing.assert_close(y, x)
        torch.testing.assert_close(mask, torch.ones_like(mask))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_forward_translation_and_inverse(self):
        self.t[:, 0, 3] = 2
        y, _ = warp_metric_bev(self.x, self.t, self.bounds)
        self.assertAlmostEqual(y[0, 0, 4, 7].item(), 1, places=5)
        restored, _ = warp_metric_bev(y, torch.linalg.inv(self.t), self.bounds)
        torch.testing.assert_close(restored, self.x)

    def test_rotation(self):
        self.t[0, :2, :2] = torch.tensor([[0., -1.], [1., 0.]])
        y, _ = warp_metric_bev(self.x, self.t, self.bounds)
        self.assertAlmostEqual(y[0, 0, 5, 4].item(), 1, places=5)

    def test_no_overlap(self):
        self.t[:, 0, 3] = 100
        y, mask = warp_metric_bev(self.x, self.t, self.bounds)
        self.assertEqual(y.sum().item(), 0)
        self.assertEqual(mask.sum().item(), 0)

    def test_recurrent_identity_legacy_parity(self):
        args = dict(feature_dim=2, hidden_dim=2, gate_dim=2,
                    state_update_mode='residual_observation', output_mode='hidden',
                    extrapolate_to_current=False, lidar_range=self.bounds)
        legacy = EctraRecurrentAlignment(args).eval()
        fixed = EctraRecurrentAlignment(dict(args, coordinate_mode='collaborator_latest')).eval()
        fixed.load_state_dict(legacy.state_dict(), strict=True)
        feats = torch.rand(4, 2, 9, 9, requires_grad=True)
        lens, times = torch.tensor([2]), torch.tensor([0., 0., -3., -6.])
        transforms = torch.eye(4).repeat(1, 2, 2, 1, 1)
        y, aux = fixed(feats, lens, times, pairwise_t_matrix=transforms)
        expected, _ = legacy(feats, lens, times)
        torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y[:2], feats[:2])
        (y.sum() + aux['ectra_motion_loss']).backward()
        for module in [fixed.motion_net, fixed.calib_net, fixed.trust_net, fixed.candidate_net]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))


class HistoryIndexTest(unittest.TestCase):
    def setUp(self):
        self.index = CausalEgoHistoryIndex([
            dict(pointcloud_path='velodyne/a.pcd', pointcloud_timestamp='1000000', batch_id='1'),
            dict(pointcloud_path='velodyne/b.pcd', pointcloud_timestamp='1100000', batch_id='1'),
            dict(pointcloud_path='velodyne/c.pcd', pointcloud_timestamp='1200000', batch_id='1'),
            dict(pointcloud_path='velodyne/d.pcd', pointcloud_timestamp='1050000', batch_id='2'),
        ], max_age_ms=100)

    def test_causal_and_sequence(self):
        self.assertEqual(self.index.select('c', 1070000), ('a', 70))
        self.assertEqual(self.index.select('b', 1200000), ('b', 100))

    def test_missing_and_tolerance(self):
        self.assertEqual(self.index.select('c', 900000), (None, None))
        self.assertEqual(self.index.select('b', 1300000), (None, 200))
        self.assertEqual(self.index.select('c', 1100000), ('b', 0))


if __name__ == '__main__':
    unittest.main()

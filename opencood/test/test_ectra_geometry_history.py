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

    def test_static_scene_uses_one_coordinate_frame(self):
        model = EctraRecurrentAlignment(dict(
            feature_dim=1, hidden_dim=2, gate_dim=2,
            coordinate_mode='collaborator_latest', lidar_range=self.bounds,
            extrapolate_to_current=False, state_update_mode='residual_observation')).eval()
        features = torch.zeros(4, 1, 9, 9)
        features[:2, 0, 4, 6] = 1  # ego sees world x=2
        features[2, 0, 4, 5] = 1   # latest CAV origin at world x=1
        features[3, 0, 4, 3] = 1   # older CAV origin at world x=3
        transforms = torch.eye(4).repeat(1, 2, 2, 1, 1)
        transforms[0, 1, 0, 0, 3] = 1
        transforms[0, 1, 1, 0, 3] = 3
        captured = []
        handle = model.motion_net.register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0].detach().clone()))
        output, _ = model(features, torch.tensor([2]), torch.tensor([0., 0., -3., -6.]),
                          pairwise_t_matrix=transforms)
        handle.remove()
        torch.testing.assert_close(captured[0][:, :1], features[2:3])
        torch.testing.assert_close(captured[0][:, 1:2], features[2:3])
        torch.testing.assert_close(output[3:], features[3:])

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

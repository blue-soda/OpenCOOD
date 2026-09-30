"""Physical alignment and message reconstruction contracts (CPU)."""
import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_geometry import encoded_grid, warp_volume
from opencood.utils import fusion_packet


class GeometryTest(unittest.TestCase):
    def test_receptive_centers(self):
        origin, step = encoded_grid([.2, .2, .1], [-100.8, -40, -3.5, 100.8, 40, 1.5])
        np.testing.assert_allclose(origin, [-100.7, -39.9, -2.65])
        np.testing.assert_allclose(step, [1.6, 1.6, 1.6])

    def test_identity_and_height_fold(self):
        x = torch.rand(2, 4, 3, 5, 7)
        y, mask = warp_volume(x, torch.eye(4).repeat(2, 1, 1), [-3., -2., -1.], [1., 1., 2.])
        self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        self.assertTrue(torch.equal(y.flatten(1, 2).reshape_as(x), x))

    def test_translation_direction(self):
        x = torch.zeros(1, 1, 5, 5, 5, dtype=torch.double)
        x[0, 0, 2, 2, 2] = 4
        matrix = torch.eye(4, dtype=torch.double)[None]
        matrix[0, 0, 3] = 1
        y, _ = warp_volume(x, matrix, [-2, -2, -2], [1, 1, 1])
        expected = torch.zeros_like(x); expected[0, 0, 2, 2, 3] = 4
        self.assertTrue(torch.allclose(y, expected, atol=1e-12))

    def test_roll_moves_height_not_just_bev(self):
        x = torch.zeros(1, 1, 5, 5, 5, dtype=torch.double)
        x[0, 0, 2, 3, 2] = 4
        matrix = torch.tensor([[[1, 0, 0, 0], [0, 0, -1, 0],
                                [0, 1, 0, 0], [0, 0, 0, 1]]], dtype=torch.double)
        y, _ = warp_volume(x, matrix, [-2, -2, -2], [1, 1, 1])
        expected = torch.zeros_like(x); expected[0, 0, 3, 2, 2] = 4
        self.assertTrue(torch.allclose(y, expected, atol=1e-12))

    def test_fractional_interpolation_gradient_and_outside(self):
        x = torch.arange(5., dtype=torch.double).reshape(1, 1, 1, 1, 5).repeat(1, 1, 3, 3, 1).requires_grad_()
        matrix = torch.eye(4, dtype=torch.double)[None]
        matrix[0, 0, 3] = .5
        y, _ = warp_volume(x, matrix, [0, 0, 0], [1, 1, 1])
        self.assertAlmostEqual(float(y[0, 0, 1, 1, 2]), 1.5)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
        matrix[0, 0, 3] = 100
        y, mask = warp_volume(x, matrix, [0, 0, 0], [1, 1, 1])
        self.assertEqual(float(y.abs().sum() + mask.sum()), 0)

    def test_packet_roundtrip_both_activations(self):
        coords = np.asarray([[0, 0, 0], [1, 2, 3], [2, 3, 4]])
        metadata = {'shape_zyx': [3, 5, 6], 'frame_id': 'test', 'source_to_ego': np.eye(4).tolist()}
        for activation in ('count4', 'relu'):
            values = np.zeros((3, 128), dtype=np.float32)
            values[0, ::3] = 4 if activation == 'count4' else .1234567
            values[1, ::5] = 1 if activation == 'count4' else 12.875
            packet = fusion_packet.encode(coords, values, metadata, activation)
            c, f, m = fusion_packet.decode(packet, activation)
            np.testing.assert_array_equal(c, coords[:2])
            np.testing.assert_array_equal(f, values[:2])
            self.assertEqual(m, metadata)


if __name__ == '__main__':
    unittest.main()

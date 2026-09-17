"""Check voxel ordering, truncation and native buffer ownership."""

import pickle
import unittest

import numpy as np

from opencood.data_utils.pre_processor.sp_voxel_preprocessor import (
    NumpyVoxelGenerator, Spconv2VoxelGenerator)

try:
    from spconv.utils import Point2VoxelCPU3d
except ImportError:
    Point2VoxelCPU3d = None


@unittest.skipIf(Point2VoxelCPU3d is None, 'spconv2 CPU interface unavailable')
class VoxelBackendTest(unittest.TestCase):
    def test_exact_output_with_truncation_and_boundaries(self):
        rng = np.random.RandomState(19)
        for cap in (1, 10, 200):
            for count in (0, 1, 10, 200, 5000):
                args = ([0.4, 0.4, 5], [-2, -2, -3, 2, 2, 2], 3, cap)
                points = rng.uniform(-3, 3, (count, 4)).astype(np.float32)
                for actual, expected in zip(
                        Spconv2VoxelGenerator(*args).generate(points),
                        NumpyVoxelGenerator(*args).generate(points)):
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual, expected)
        points = np.array([[-2, -2, -3, 1], [2, 2, 2, 2],
                           [0, 0, 0, 3], [0, 0, 0, 4]], dtype=np.float32)
        args = ([0.4, 0.4, 5], [-2, -2, -3, 2, 2, 2], 1, 10)
        for a, b in zip(Spconv2VoxelGenerator(*args).generate(points),
                        NumpyVoxelGenerator(*args).generate(points)):
            np.testing.assert_array_equal(a, b)

    def test_history_buffers_and_spawn_serialization(self):
        generator = Spconv2VoxelGenerator([0.4, 0.4, 5],
                                          [-2, -2, -3, 2, 2, 2], 3, 10)
        first = np.array([[0, 0, 0, 1]], dtype=np.float32)
        previous = generator.generate(first)
        saved = tuple(x.copy() for x in previous)
        generator.generate(np.array([[1, 1, 0, 9]], dtype=np.float32))
        for actual, expected in zip(previous, saved):
            np.testing.assert_array_equal(actual, expected)
        restored = pickle.loads(pickle.dumps(generator))
        for actual, expected in zip(restored.generate(first), saved):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

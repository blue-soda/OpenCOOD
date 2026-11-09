"""Rotations must not change IoU when the same rigid transform is applied to both boxes."""
import numpy as np
from opencood.utils.eval_oriented_3d_utils import pairwise_oriented_iou3d


def test_cuboid_volume_under_tilt():
    a = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)[None]
    c = np.sqrt(.5)
    tilt = np.array([[1,0,0],[0,c,-c],[0,c,c]])
    b = np.concatenate([a, a+[0,0,1], a+[0,0,2]])
    np.testing.assert_allclose(pairwise_oriented_iou3d(a @ tilt, b @ tilt), [[1,1/3,0]], atol=1e-8)
    # The cross-section perpendicular to x is the intersection of two equal
    # concentric squares rotated through 45 degrees: IoU = 1/sqrt(2).
    np.testing.assert_allclose(pairwise_oriented_iou3d(a, a @ tilt), [[c]], atol=1e-8)
    np.testing.assert_allclose(pairwise_oriented_iou3d(a @ tilt, a), [[c]], atol=1e-8)

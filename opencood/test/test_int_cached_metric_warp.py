import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cached_metric_warp import CachedMetricWarp


class CachedMetricWarpTests(unittest.TestCase):
    def test_exact_nearest_geometry_boundary_and_cache(self):
        torch.manual_seed(303)
        bounds = [-17,-9,-2,31,7,2]
        fast = CachedMetricWarp(bounds)
        identity = np.eye(4)
        for dtype in (torch.float32, torch.float64):
            for h,w in ((1,1),(7,19),(8,12)):
                value = torch.randn(1,9,h,w,dtype=dtype)
                before = value.clone()
                for shift,yaw in ((0,0),(48/w*.5,0),(48/w*.501,0),(1.7,.13),(1000,0)):
                    pose = np.eye(4); pose[0,3] = shift
                    pose[0,0] = pose[1,1] = np.cos(yaw)
                    pose[0,1] = -np.sin(yaw); pose[1,0] = np.sin(yaw)
                    expected, valid = warp_bev(value,identity,pose,bounds)
                    actual, actual_valid = fast(value,identity,pose,bounds)
                    self.assertTrue(torch.equal(expected,actual))
                    self.assertTrue(torch.equal(valid,actual_valid))
                    size = fast.geometry_bytes
                    fast(value,identity,pose,bounds)
                    self.assertEqual(size,fast.geometry_bytes)
                self.assertTrue(torch.equal(before,value))
        self.assertLessEqual(len(fast.geometry),4)


if __name__ == '__main__': unittest.main()

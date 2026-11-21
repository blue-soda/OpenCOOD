import unittest

import torch

from opencood.models.sub_modules.int_compact_middle_update import supports
from opencood.models.sub_modules.int_protected_middle import protected_mask,sparse_returns


class ProtectionTest(unittest.TestCase):
    def test_same_incremental_work_at_boundaries_and_partial_tiles(self):
        gen=torch.Generator().manual_seed(14)
        base=torch.rand((1,1,27,39),generator=gen)>.1
        silent=base & (torch.rand(base.shape,generator=gen)>.15)
        score=torch.rand(base.shape,generator=gen)
        for frame in range(8):
            costs=[]
            for signal in (score,1-score,torch.zeros_like(score)):
                mask,info=protected_mask(base,silent,signal,frame)
                self.assertFalse((mask&~silent).any())
                old=base&~silent
                diff=[int((a & ~b).sum()) for a,b in zip(supports(old|mask),supports(old))]
                self.assertEqual(diff,info['increments'])
                costs.append(diff)
            self.assertEqual(costs[0],costs[1])
            self.assertEqual(costs[0],costs[2])

    def test_voxel_point_counts_are_not_occupancy_counts(self):
        coords=torch.tensor([[0,0,4,4],[0,0,4,5],[0,0,16,16]])
        sparse=sparse_returns(dict(voxel_coords=coords,voxel_num_points=torch.tensor([12,10,3])),(20,20))
        self.assertFalse(sparse[0,0,1,1])
        self.assertTrue(sparse[0,0,4,4])

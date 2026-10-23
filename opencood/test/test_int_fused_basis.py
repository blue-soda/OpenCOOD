import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache
from opencood.models.sub_modules.int_fused_basis import fuse_spatial_basis


class FusedBasisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.original = nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval()
        self.stage = SpatialBasisStage(self.original, 9).eval()
        self.fused = fuse_spatial_basis(self.stage)
        self.x = torch.rand(1, 16, 12, 20, dtype=torch.double)

    def test_dense_and_partial_equivalence_and_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.fused.parameters()), sum(p.numel() for p in self.original.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.fused.parameters()))
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.stage(self.x), self.fused(self.x), atol=1e-10, rtol=1e-10))
            current = self.stage.observe(self.x)
            old = [torch.rand_like(current) for _ in range(4)]
            for fraction in (0., .5, 1.):
                mask = torch.rand(1, 1, 12, 20) < fraction
                a, _, ma = self.stage.selected_from_stem(current, old, mask)
                b, maps, mb = self.fused.selected_from_stem(current, old, mask)
                self.assertEqual(ma, mb)
                self.assertTrue(torch.allclose(a, b, atol=1e-10, rtol=1e-10))
                for actual, cached in zip(maps, old):
                    self.assertTrue(torch.equal(actual.masked_select(~mask), cached.masked_select(~mask)))

    def test_recursive_cache_unchanged(self):
        left = SpatialBasisCache(self.stage, [-10,-6,-2,10,6,2], .5)
        right = SpatialBasisCache(self.fused, [-10,-6,-2,10,6,2], .5)
        ls = rs = None
        with torch.no_grad():
            for frame in range(5):
                pose = np.eye(4); pose[0,3] = frame*.25
                meta = dict(segment='a', timestamp_us=frame*100000, pose=pose)
                x = self.x.clone(); x[:,:,4:6,8:10] += frame*.2
                a,ls,ia = left.step(x,ls,meta); b,rs,ib = right.step(x,rs,meta)
                self.assertTrue(torch.allclose(a,b,atol=1e-10,rtol=1e-10))
                self.assertEqual(ia['spatial_selected_cells'],ib['spatial_selected_cells'])
                self.assertEqual(ia['spatial_extra_linear_macs'],ib['spatial_extra_linear_macs'])


if __name__ == '__main__': unittest.main()

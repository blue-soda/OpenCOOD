import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache
from opencood.models.sub_modules.int_fused_basis import fuse_spatial_basis
from opencood.models.sub_modules.int_selected_conv import prepare_selected_neighborhoods, prepared_conv3x3_rows, selected_conv3x3_rows


class FusedBasisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.original = nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval()
        self.stage = SpatialBasisStage(self.original, 9).eval()
        self.fused = fuse_spatial_basis(self.stage)
        self.prepared = fuse_spatial_basis(self.stage, prepared=True)
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
                c, prepared_maps, mc = self.prepared.selected_from_stem(current, old, mask)
                self.assertEqual(mc, mb)
                self.assertTrue(torch.allclose(c, b, atol=1e-10, rtol=1e-10))
                self.assertEqual(sum(p.numel() for p in self.prepared.parameters()), sum(p.numel() for p in self.fused.parameters()))
                for actual, cached in zip(maps, old):
                    self.assertTrue(torch.equal(actual.masked_select(~mask), cached.masked_select(~mask)))
                for actual, cached in zip(prepared_maps, old):
                    self.assertTrue(torch.equal(actual.masked_select(~mask), cached.masked_select(~mask)))

    def test_recursive_cache_unchanged(self):
        left = SpatialBasisCache(self.stage, [-10,-6,-2,10,6,2], .5)
        right = SpatialBasisCache(self.fused, [-10,-6,-2,10,6,2], .5)
        optimized = SpatialBasisCache(self.prepared, [-10,-6,-2,10,6,2], .5)
        ls = rs = os = None
        with torch.no_grad():
            for frame in range(5):
                pose = np.eye(4); pose[0,3] = frame*.25
                meta = dict(segment='a', timestamp_us=frame*100000, pose=pose)
                x = self.x.clone(); x[:,:,4:6,8:10] += frame*.2
                a,ls,ia = left.step(x,ls,meta); b,rs,ib = right.step(x,rs,meta)
                self.assertTrue(torch.allclose(a,b,atol=1e-10,rtol=1e-10))
                self.assertEqual(ia['spatial_selected_cells'],ib['spatial_selected_cells'])
                self.assertEqual(ia['spatial_extra_linear_macs'],ib['spatial_extra_linear_macs'])
                c,os,ic = optimized.step(x,os,meta)
                self.assertTrue(torch.allclose(b,c,atol=1e-10,rtol=1e-10))
                self.assertEqual(ib['spatial_selected_cells'],ic['spatial_selected_cells'])
                self.assertEqual(ib['spatial_extra_linear_macs'],ic['spatial_extra_linear_macs'])

    def test_prepared_boundaries_chunks_and_counts(self):
        conv = nn.Conv2d(3, 5, 3, padding=1).double().eval()
        x = torch.rand(1, 3, 65, 81, dtype=torch.double)
        weight = conv.weight.detach().permute(0,2,3,1).reshape(5,-1).contiguous()
        with torch.no_grad():
            for fraction in (0., .4, 1.):
                mask = torch.rand(1,1,65,81) < fraction
                plan = prepare_selected_neighborhoods(mask)
                rows, ids, macs = prepared_conv3x3_rows(conv, weight, x, plan, chunk_size=1024)
                ref, ref_ids, ref_macs = selected_conv3x3_rows(conv, x, mask)
                self.assertTrue(torch.equal(ids,ref_ids))
                self.assertEqual(macs,ref_macs)
                self.assertTrue(torch.allclose(rows,ref,atol=1e-12,rtol=1e-12))
                expected = conv(x)[0].permute(1,2,0).reshape(-1,5)[ids]
                self.assertTrue(torch.allclose(rows,expected,atol=1e-12,rtol=1e-12))


if __name__ == '__main__': unittest.main()

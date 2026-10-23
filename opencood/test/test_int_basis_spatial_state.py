import copy
import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisConv, SpatialBasisStage


class BasisStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.source = nn.Sequential(*[BasicBlock(12, 12) for _ in range(3)]).double().eval()
        # Nontrivial fixed BN statistics test residual/nonlinearity placement.
        with torch.no_grad():
            for m in self.source.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.uniform_(-.3, .3); m.running_var.uniform_(.7, 1.3)
        self.x = torch.randn(1, 12, 7, 11, dtype=torch.double)

    def test_full_rank_preserves_mature_function_and_weights(self):
        original = copy.deepcopy(self.source.state_dict())
        model = SpatialBasisStage(self.source, 9).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(self.x), self.source(self.x), atol=1e-10, rtol=1e-10))
        self.assertTrue(all(torch.equal(v, self.source.state_dict()[k]) for k, v in original.items()))
        self.assertTrue(all(abs(e-1) < 1e-12 for e in model.retained_weight_energy))

    def test_selected_mixed_cache_matches_masked_dense_and_counts(self):
        model = SpatialBasisStage(self.source, 6).eval()
        with torch.no_grad():
            current = model.observe(self.x)
            old = [torch.rand_like(current) for _ in range(4)]
            mask = torch.rand(1, 1, 7, 11) > .65
            sparse, states, macs = model.selected_from_stem(current, old, mask)
            dense, reference, _ = model.selected_from_stem(current, old, mask, 'masked_dense')
            self.assertTrue(torch.allclose(sparse, dense, atol=1e-10, rtol=1e-10))
            self.assertEqual(macs, int(mask.sum())*4*12*12*6)
            for actual, expected, cached in zip(states, reference, old):
                self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))
                self.assertTrue(torch.equal(actual.masked_select(~mask), cached.masked_select(~mask)))

    def test_all_and_empty_masks(self):
        model = SpatialBasisStage(self.source, 9).eval()
        with torch.no_grad():
            current = model.observe(self.x); expected, old = model.dense_from_stem(current)
            empty = torch.zeros(1, 1, 7, 11, dtype=torch.bool)
            result, states, macs = model.selected_from_stem(current, old, empty)
            self.assertTrue(torch.equal(result, old[-1])); self.assertEqual(macs, 0)
            result, _, macs = model.selected_from_stem(current, old, ~empty)
            self.assertTrue(torch.allclose(result, expected, atol=1e-10, rtol=1e-10))
            self.assertEqual(macs, 7*11*4*12*12*9)

    def test_round_trip_and_reduced_rank(self):
        model = SpatialBasisStage(self.source, 6).eval()
        other = SpatialBasisStage(self.source, 6).eval(); other.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(all(0 < energy < 1 for energy in model.retained_weight_energy))
        with torch.no_grad(): self.assertTrue(torch.equal(model(self.x), other(self.x)))

    def test_rejects_unsupported_geometry_and_training(self):
        with self.assertRaises(ValueError): SpatialBasisConv(nn.Conv2d(12, 12, 3, stride=2, padding=1), 9)
        model = SpatialBasisStage(self.source, 9).eval()
        with self.assertRaises(ValueError): model.selected_from_stem(self.x, [self.x]*4, torch.ones(1, 1, 7, 11, dtype=torch.bool))


if __name__ == '__main__': unittest.main()

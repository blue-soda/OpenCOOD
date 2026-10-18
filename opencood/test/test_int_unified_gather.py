import unittest
import torch
from opencood.models.sub_modules.int_unified_bev import ResidualStateUpdate
from opencood.models.sub_modules.int_unified_gather import gathered_update


class GatheredUpdateTests(unittest.TestCase):
    def test_matches_dense_at_boundaries_and_disconnected_cells(self):
        torch.manual_seed(903)
        module = ResidualStateUpdate(evidence=6, channels=8).eval().double()
        # Nontrivial biases/norm affine values expose accidental padded activations.
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.uniform_(-.4, .4)
            for height, width in [(1, 1), (2, 5), (9, 13)]:
                x = torch.randn(1, 15, height, width, dtype=torch.double)
                x[:, -1:] = torch.rand_like(x[:, -1:]) > .3
                old = x[:, 6:14]
                for kind in ['empty', 'all', 'scattered', 'edges']:
                    mask = torch.zeros(1, 1, height, width, dtype=torch.bool)
                    if kind == 'all': mask.fill_(True)
                    if kind == 'scattered': mask = torch.rand_like(mask.float()) > .65
                    if kind == 'edges':
                        mask[..., 0, :] = True
                        mask[..., -1, -1] = True
                    expected = torch.where(mask, module(x), old)
                    actual = gathered_update(module, x, mask)
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10), kind)
                    self.assertTrue(torch.equal(actual[~mask.expand_as(actual)], old[~mask.expand_as(old)]))

    def test_rejects_training(self):
        module = ResidualStateUpdate(evidence=6, channels=8)
        x = torch.randn(1, 15, 3, 5)
        mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
        with torch.no_grad(), self.assertRaises(ValueError):
            gathered_update(module, x, mask)
        module.eval()
        with self.assertRaises(ValueError): gathered_update(module, x, mask)

    def test_normalized_shortcut_same_parameters_and_gather_equivalence(self):
        torch.manual_seed(307);plain=ResidualStateUpdate(evidence=6,channels=8).eval().double()
        torch.manual_seed(307);norm=ResidualStateUpdate(evidence=6,channels=8,normalize_current=True).eval().double()
        self.assertEqual(list(plain.state_dict()),list(norm.state_dict()))
        for k in plain.state_dict():self.assertTrue(torch.equal(plain.state_dict()[k],norm.state_dict()[k]))
        with torch.no_grad():
            x=torch.randn(1,15,9,13,dtype=torch.double)*1000;x[:,-1:]=1
            value=norm.current_value(x[:,:6])
            self.assertLess(float(value.mean(1).abs().max()),1e-10)
            self.assertTrue(torch.allclose(value.square().mean(1),torch.ones_like(value[:,0]),atol=1e-6))
            mask=torch.rand(1,1,9,13)>.6
            self.assertTrue(torch.allclose(gathered_update(norm,x,mask),torch.where(mask,norm(x),x[:,6:14]),atol=1e-9,rtol=1e-9))


if __name__ == '__main__': unittest.main()

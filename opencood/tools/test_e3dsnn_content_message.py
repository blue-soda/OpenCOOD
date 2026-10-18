"""Content protocol invariants, not detection quality or compression claims."""
import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_content_message import CountContentBottleneck
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume


class ContentMessageTest(unittest.TestCase):
    def test_binary_events_recover_full_counts(self):
        for family in ('count3', 'ann2bit'):
            model = CountContentBottleneck(4, 4, family).double()
            x = torch.tensor([0., 4./3., 8./3., 4.], dtype=torch.double).reshape(1, 4, 1, 1, 1)
            content = model.encode(x)
            events = [content.event(k) for k in (1, 2, 3)]
            for event in events:
                self.assertTrue(torch.all((event == 0) | (event == 1)))
            self.assertTrue(torch.equal(sum(events), content.counts))
            for k in (1, 2, 3):
                torch.testing.assert_close(sum(events[:k]) * content.scales, content.prefix(k))
            self.assertTrue(torch.equal(content.prefix(0), torch.zeros_like(content.counts)))

    def test_empty_voxels_and_frame_scales(self):
        for family in ('count3', 'ann2bit'):
            model = CountContentBottleneck(4, 2, family)
            x = torch.zeros(1, 4, 1, 1, 3)
            x[:, :, :, :, 1:] = torch.tensor([1., 4.])
            content = model.encode(x)
            for k in (1, 2, 3):
                self.assertTrue(torch.equal(content.prefix(k)[..., 0], torch.zeros(1, 2, 1, 1)))
            if family == 'ann2bit':
                self.assertTrue(torch.all(content.scales == 4./3.))
            empty = model.encode(torch.zeros_like(x))
            self.assertTrue(torch.all(empty.scales == 1))
            self.assertTrue(torch.all(model.decode(empty.prefix(3)) == 0))

    def test_learning_reaches_both_matrices(self):
        for family in ('count3', 'ann2bit'):
            model = CountContentBottleneck(4, 2, family).double()
            content = model.encode(torch.full((1, 4, 1, 2, 2), 1.2, dtype=torch.double))
            loss = sum(model.decode(content.prefix(k)).square().mean() for k in (1, 2, 3))
            loss.backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_align_then_expand_and_accumulate(self):
        torch.manual_seed(3)
        model = CountContentBottleneck(4, 2).double()
        # Nontrivial signed learned decoding remains linear; it is local current.
        with torch.no_grad(): model.expand.weight.normal_()
        content = model.encode(torch.rand(1, 4, 2, 3, 4, dtype=torch.double) * 4)
        transform = torch.eye(4, dtype=torch.double)[None]
        transform[0, 0, 3] = .35
        transform[0, 1, 3] = -.2
        def warp(x):
            return warp_volume(x, transform, (0., 0., 0.), (1., 1., 1.), geometry_dtype=torch.float64)[0]
        full = content.prefix(3)
        torch.testing.assert_close(model.decode(warp(full)), warp(model.decode(full)))
        accumulated = sum(warp(content.event(k) * content.scales) for k in (1, 2, 3))
        torch.testing.assert_close(model.decode(accumulated), model.decode(warp(full)))

    def test_basis_identity_changes_with_content_or_decoder(self):
        model = CountContentBottleneck(4, 2)
        before = model.basis_digest()
        with torch.no_grad(): model.project.weight[0, 0] += .01
        self.assertNotEqual(before, model.basis_digest())
        before = model.basis_digest()
        with torch.no_grad(): model.expand.weight[0, 0] += .01
        self.assertNotEqual(before, model.basis_digest())
        self.assertEqual(model.basis_digest(), model.basis_digest())

    def test_contract_shape_and_indices(self):
        model = CountContentBottleneck(4, 2)
        for x in (torch.zeros(2, 4, 1, 1, 1), torch.zeros(1, 3, 1, 1, 1),
                  -torch.ones(1, 4, 1, 1, 1), torch.full((1, 4, 1, 1, 1), float('nan'))):
            with self.assertRaises(ValueError): model.encode(x)
        content = model.encode(torch.ones(1, 4, 1, 1, 1))
        for index in (True, -1, 4, 1.5):
            with self.assertRaises(ValueError): content.prefix(index)
        with self.assertRaises(ValueError): content.event(0)
        with self.assertRaises(ValueError): model.decode(torch.zeros(1, 3, 1, 1, 1))


if __name__ == '__main__': unittest.main()

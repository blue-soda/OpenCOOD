import unittest
import torch
from opencood.models.sub_modules.e3dsnn_highres_bev import LocalCountDetail, upsample_bev


class HighResolutionTest(unittest.TestCase):
    def test_width_and_receptive_field_preserve_count_interface(self):
        torch.set_num_threads(1)
        modules = [LocalCountDetail(), LocalCountDetail(256, 1), LocalCountDetail(128, 2)]
        self.assertEqual(sum(p.numel() for p in modules[0].parameters()),
                         sum(p.numel() for p in modules[2].parameters()))
        for key, value in modules[0].state_dict().items():
            self.assertTrue(torch.equal(value, modules[2].state_dict()[key]))
        self.assertGreater(sum(p.numel() for p in modules[1].parameters()),
                           sum(p.numel() for p in modules[0].parameters()))
        for module in modules:
            counts = torch.randint(0, 5, (2, 64, 12, 16)).float().requires_grad_()
            self.assertEqual(module(counts).shape, (2, 256, 12, 16))
            self.assertEqual(module(counts).abs().sum().item(), 0)
            with torch.no_grad(): module.output.weight.normal_(0, .01)
            out = module(counts); out.square().mean().backward()
            self.assertTrue(torch.isfinite(counts.grad).all())
            self.assertGreater(counts.grad.abs().sum().item(), 0)
            self.assertLessEqual(out.abs().max().item(), 1)

    def test_zero_initialization_then_gradient_and_bound(self):
        torch.set_num_threads(1)
        module = LocalCountDetail()
        counts = torch.randint(0, 5, (2, 64, 12, 16)).float().requires_grad_()
        bev = torch.randn(2, 256, 6, 8)
        delta = module(counts)
        self.assertTrue(torch.equal(upsample_bev(bev) + delta, upsample_bev(bev)))
        (delta * torch.randn_like(delta)).sum().backward()
        self.assertGreater(module.output.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():module.output.weight.add_(module.output.weight.grad * .01)
        module.zero_grad(); counts.grad = None
        out = module(counts); (out * torch.randn_like(out)).sum().backward()
        self.assertGreater(counts.grad.abs().sum().item(), 0)
        self.assertLessEqual(module(counts.detach() * 1e6).abs().max().item(), 1)
        self.assertTrue(torch.equal(module(torch.zeros_like(counts)), torch.zeros_like(out)))


if __name__ == '__main__':unittest.main()

import unittest
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3


class SelectedConvTests(unittest.TestCase):
    def test_masked_dense_equivalence_boundaries_and_chunks(self):
        torch.manual_seed(303)
        for height, width in ((1, 1), (1, 7), (7, 1), (9, 13)):
            for bias in (False, True):
                conv = torch.nn.Conv2d(3, 5, 3, bias=bias).double().eval()
                x = torch.randn(1, 3, height, width, dtype=torch.double)
                cache = torch.randn(1, 5, height, width, dtype=torch.double)
                original_x, original_cache = x.clone(), cache.clone()
                for mode in ('none', 'all', 'random'):
                    mask = torch.rand(1, 1, height, width) > .5
                    if mode != 'random':
                        mask.fill_(mode == 'all')
                    with torch.no_grad():
                        actual, macs = selected_conv3x3(conv, x, cache, mask, chunk_size=3)
                        dense = conv(F.pad(x, (1, 1, 1, 1)))
                    expected = torch.where(mask, dense, cache)
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
                    self.assertEqual(macs, int(mask.sum()) * 3 * 5 * 9)
                    self.assertTrue(torch.equal(actual[~mask.expand_as(actual)], cache[~mask.expand_as(cache)]))
                    self.assertTrue(torch.equal(x, original_x))
                    self.assertTrue(torch.equal(cache, original_cache))

    def test_internal_padding_and_channels_last_input(self):
        torch.manual_seed(304)
        conv = torch.nn.Conv2d(4, 6, 3, padding=1).eval()
        x = torch.randn(1, 4, 6, 9).to(memory_format=torch.channels_last)
        mask = torch.ones(1, 1, 6, 9, dtype=torch.bool)
        cache = torch.zeros(1, 6, 6, 9)
        with torch.no_grad():
            actual, _ = selected_conv3x3(conv, x, cache, mask, chunk_size=7)
            expected = conv(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_training_and_unsupported_operator_rejected(self):
        x = torch.zeros(1, 3, 5, 7)
        cache = torch.zeros(1, 4, 5, 7)
        mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
        conv = torch.nn.Conv2d(3, 4, 3).eval()
        with self.assertRaises(ValueError):
            selected_conv3x3(conv, x, cache, mask)
        with torch.no_grad():
            conv.train()
            with self.assertRaises(ValueError):
                selected_conv3x3(conv, x, cache, mask)
            conv = torch.nn.Conv2d(3, 4, 3, stride=2).eval()
            with self.assertRaises(ValueError):
                selected_conv3x3(conv, x, cache, mask)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_transposed_integer_events import transposed_integer_events


class TransposedIntegerEventTests(unittest.TestCase):
    def test_signed_sparse_dense_zero_geometry_and_exact_counts(self):
        torch.manual_seed(303)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for k in (1, 2, 4):
                conv = nn.ConvTranspose2d(3, 5, k, stride=k).to(device)
                weights = torch.randint(-32767, 32768, conv.weight.shape, dtype=torch.int16, device=device)
                for h, w in ((1, 1), (3, 4), (4, 3)):
                    for mode in ('zero', 'sparse', 'dense'):
                        x = torch.randint(-15, 16, (1, 3, h, w), dtype=torch.int64, device=device)
                        if mode == 'zero': x.zero_()
                        if mode == 'sparse': x *= torch.rand(x.shape, device=device) < .3
                        if mode == 'dense': x[x == 0] = 1
                        ref = F.conv_transpose2d(x.double(), weights.double(), stride=k).to(torch.int64)
                        with torch.no_grad(), patch('torch.nn.functional.conv_transpose2d', side_effect=AssertionError('Dense call')):
                            for chunk in (1, 7, 32768):
                                y, stats = transposed_integer_events(conv, x, weights, chunk)
                                self.assertTrue(torch.equal(y, ref))
                                self.assertEqual(y.dtype, torch.int64)
                                self.assertEqual(stats['executed_macs'], int((x != 0).sum()) * 5 * k*k)
                                self.assertEqual(stats['unit_spike_ac'], int(x.abs().sum()) * 5 * k*k)
                                self.assertEqual(stats['dense_macs'], h*w*3*5*k*k)
                                self.assertEqual(stats['dense_conv_calls'], 0)

    def test_own_state_delta_refresh_and_restored_state(self):
        torch.manual_seed(304)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for k in (1, 2, 4):
                conv = nn.ConvTranspose2d(2, 3, k, stride=k).to(device)
                weights = torch.randint(-37, 38, conv.weight.shape, dtype=torch.int16, device=device)
                old = acc = None
                with torch.no_grad():
                    for step in range(8):
                        x = torch.randint(-8, 9, (1, 2, 3, 4), dtype=torch.int64, device=device)
                        reset = step in (0, 4, 7)
                        update, _ = transposed_integer_events(conv, x if reset else x-old, weights, 3)
                        acc = update if reset else acc + update
                        ref = F.conv_transpose2d(x.double(), weights.double(), stride=k).to(torch.int64)
                        self.assertTrue(torch.equal(acc, ref)); old = x.clone()
                        # Copy through CPU to model a restored integer execution cache.
                        old, acc = old.cpu().clone().to(device), acc.cpu().clone().to(device)

    def test_invalid_geometry_type_and_bounded_arithmetic(self):
        conv = nn.ConvTranspose2d(2, 3, 2, stride=2)
        x = torch.ones(1, 2, 3, 4, dtype=torch.int64)
        w = torch.ones_like(conv.weight, dtype=torch.int16)
        with self.assertRaises(ValueError): transposed_integer_events(conv, x, w)
        with torch.no_grad():
            for bad in (x.float(), x.expand(2, -1, -1, -1), x*2**31, x*(-2**63)):
                with self.assertRaises(ValueError): transposed_integer_events(conv, bad, w)
            with self.assertRaises(ValueError): transposed_integer_events(conv, x, w, 0)
            with self.assertRaises(ValueError): transposed_integer_events(conv, x, w.long())
            conv.padding = (1, 1)
            with self.assertRaises(ValueError): transposed_integer_events(conv, x, w)
            wide = nn.ConvTranspose2d(129, 1, 1)
            with self.assertRaises(ValueError): transposed_integer_events(wide, torch.full((1,129,1,1), 2**31-1, dtype=torch.int64), torch.full_like(wide.weight, 32767, dtype=torch.int16))


if __name__ == '__main__': unittest.main()

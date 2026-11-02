import unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_strided_integer_events import strided_integer_events
from opencood.models.sub_modules.int_integer_event_reference import dense_integer_reference, folded_integer_weights, integer_input


class StridedIntegerEventTests(unittest.TestCase):
    def test_geometry_signed_zero_dense_and_chunks(self):
        torch.manual_seed(303)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for h, w in ((5, 6), (6, 5), (1, 1)):
                for kernel in (1, 3):
                    for stride in (1, 2):
                        conv = nn.Conv2d(2, 3, kernel, stride=stride, padding=kernel // 2).to(device)
                        weight = torch.randint(-31, 32, conv.weight.shape, dtype=torch.int16, device=device)
                        for mode in ('signed', 'zero', 'dense'):
                            values = torch.randint(-5, 6, (1, 2, h, w), dtype=torch.int64, device=device)
                            if mode == 'zero': values.zero_()
                            if mode == 'dense': values.fill_(1)
                            calls = []; handle = conv.register_forward_hook(lambda *_: calls.append(1))
                            with torch.no_grad():
                                reference, _ = dense_integer_reference(conv, values, weight)
                                for chunk in (1, 32768):
                                    result, stats = strided_integer_events(conv, values, weight, chunk)
                                    self.assertTrue(torch.equal(result, reference))
                                    oh, ow = result.shape[-2:]
                                    expected = 0
                                    # Output-centric enumeration, independent of event index arithmetic.
                                    for oy in range(oh):
                                        for ox in range(ow):
                                            for ky in range(kernel):
                                                for kx in range(kernel):
                                                    iy, ix = oy * stride - kernel // 2 + ky, ox * stride - kernel // 2 + kx
                                                    if 0 <= iy < h and 0 <= ix < w:
                                                        expected += int((values[0, :, iy, ix] != 0).sum())
                                    self.assertEqual(stats['executed_macs'], expected * 3)
                                    self.assertEqual(stats['dense_macs'], oh * ow * 2 * 3 * kernel * kernel)
                            handle.remove(); self.assertEqual(calls, [])

    def test_residual_block_own_state_stream_and_refresh(self):
        torch.manual_seed(303)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            convs = [nn.Conv2d(2, 3, 3, stride=2, padding=1), nn.Conv2d(3, 3, 3, padding=1), nn.Conv2d(2, 3, 1, stride=2)]
            convs = [c.to(device) for c in convs]
            qs = [torch.full((c.in_channels,), .125, device=device) for c in convs]
            weights = [folded_integer_weights(c, q) for c, q in zip(convs, qs)]
            bns = [nn.BatchNorm2d(3).to(device).eval() for _ in convs]
            with torch.no_grad():
                for bn in bns:
                    bn.running_mean.uniform_(-.5, .5); bn.running_var.uniform_(.5, 2); bn.weight.uniform_(.5, 1.5); bn.bias.uniform_(-.5, .5)
                old = [None] * 3; acc = [None] * 3
                for step in range(7):
                    source = torch.randn(1, 2, 7, 8, device=device)
                    paths = [[], []]
                    for arm in (0, 1):
                        for i, (conv, q, (weight, scale), bn) in enumerate(zip(convs, qs, weights, bns)):
                            x = source if i in (0, 2) else torch.relu(paths[arm][0])
                            values = integer_input(x, q)
                            reference, _ = dense_integer_reference(conv, values, weight)
                            if arm == 0: actual = reference
                            else:
                                if step in (0, 3): acc[i] = reference
                                else:
                                    update, _ = strided_integer_events(conv, values - old[i], weight, chunk=3)
                                    acc[i] = acc[i] + update
                                self.assertTrue(torch.equal(acc[i], reference)); old[i] = values; actual = acc[i]
                            decoded = (actual.double() * scale.reshape(1, -1, 1, 1)).float() + conv.bias.reshape(1, -1, 1, 1)
                            paths[arm].append(bn(decoded))
                    self.assertTrue(torch.equal(torch.relu(paths[0][1] + paths[0][2]), torch.relu(paths[1][1] + paths[1][2])))

    def test_invalid_geometry_range_and_training(self):
        conv = nn.Conv2d(2, 3, 3, stride=2, padding=1)
        values = torch.ones(1, 2, 3, 4, dtype=torch.int64)
        weight = torch.ones_like(conv.weight, dtype=torch.int16)
        with self.assertRaises(ValueError): strided_integer_events(conv, values, weight)
        with torch.no_grad():
            with self.assertRaises(ValueError): strided_integer_events(conv, values * 2**32, weight)
            with self.assertRaises(ValueError): strided_integer_events(conv, values, weight, chunk=0)
            conv.stride = (3, 3)
            with self.assertRaises(ValueError): strided_integer_events(conv, values, weight)


if __name__ == '__main__': unittest.main()

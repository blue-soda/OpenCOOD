import unittest
import torch
from opencood.models.sub_modules.int_event_cost import spatial_event_cost


class EventCostTest(unittest.TestCase):
    def test_enumerated_routing_matches_rectangular_and_degenerate_shapes(self):
        # Independent output-centric enumeration, not the input fanout formula.
        for h, w in ((1, 1), (1, 5), (4, 1), (2, 2), (3, 5)):
            x = (torch.arange(2*2*h*w).reshape(2, 2, h, w)%13).float()/8
            x.flatten()[0] = -1
            scales = torch.ones(1, 2, 1, 1)
            for initial in (0, 32768):
                result = spatial_event_cost(x, scales, 8, initial, 3)
                ac = tail = sparse = 0
                for batch in range(2):
                    for channel in range(2):
                        for oy in range(h):
                            for ox in range(w):
                                for ky in range(-1, 2):
                                    for kx in range(-1, 2):
                                        iy, ix = oy+ky, ox+kx
                                        if not (0 <= iy < h and 0 <= ix < w):
                                            continue
                                        value = max(float(x[batch, channel, iy, ix]), 0)
                                        if value > 1:
                                            tail += 3
                                        else:
                                            count = (int(value*65536)*8+initial)//65536
                                            ac += count*3
                                            sparse += (count > 0)*3
                self.assertEqual((ac, tail, sparse), (result['event_ac_exact'], result['analog_mac_exact'], result['sparse_count_mac_exact']))

    def test_channel_scales_and_half_threshold(self):
        x = torch.tensor([[[[0.0625, 1., 1.1]], [[0.125, 2., 2.2]]]])
        r = spatial_event_cost(x, torch.tensor([1., 2.]).view(1, 2, 1, 1), 8, 32768, 4)
        self.assertEqual(r['spike_count'], 18)
        self.assertEqual(r['event_ac_exact'], 208)
        self.assertEqual(r['analog_mac_exact'], 16)

    def test_nonfinite_or_zero_scale_rejected(self):
        for scales in (torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), float('nan'))):
            with self.assertRaises(ValueError):
                spatial_event_cost(torch.zeros(1, 1, 1, 1), scales, 8, 0, 2)


if __name__ == '__main__':
    unittest.main()

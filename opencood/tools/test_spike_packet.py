"""Small codec contract tests; run without torch or GPU."""
import unittest
import numpy as np
from opencood.utils.spike_packet import encode, decode


class PacketTests(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.RandomState(4)
        for n in (0, 1, 17):
            for c in (1, 13, 128):
                coords = np.column_stack([np.arange(n), np.zeros(n), np.full(n, 65535)])
                values = rng.randint(0, 5, (n, c)).astype(np.float32)
                values[:1] = 0
                keep = np.any(values, axis=1)
                metadata = {'frame': 'test', 'shape': [65536, 1, 65536]}
                packets = [encode(coords, values, metadata, mode) for mode in ('dense3', 'mask3', 'auto')]
                self.assertEqual(len(packets[-1]), min(map(len, packets[:2])))
                for packet in packets:
                    restored_coords, restored, meta = decode(packet)
                    np.testing.assert_array_equal(restored_coords, coords[keep])
                    np.testing.assert_array_equal(restored, values[keep])
                    self.assertEqual(meta, metadata)

    def test_bad_input(self):
        coords = np.zeros((1, 3))
        for value in (-1., .1, 5., np.nan, np.inf):
            with self.assertRaises(ValueError):
                encode(coords, np.array([[value]]), {})
        for invalid in (np.array([[-1, 0, 0]]), np.array([[65536, 0, 0]]), np.array([[.5, 0, 0]])):
            with self.assertRaises(ValueError):
                encode(invalid, np.ones((1, 1)), {})
        with self.assertRaises(ValueError):
            encode(np.zeros((2, 3)), np.ones((2, 1)), {})

    def test_damage(self):
        packet = encode(np.zeros((1, 3)), np.ones((1, 128)), {'frame': 1})
        for damaged in (packet[:-1], packet + b'x', packet[:25] + bytes([packet[25] ^ 1]) + packet[26:]):
            with self.assertRaises(ValueError):
                decode(damaged)


if __name__ == '__main__':
    unittest.main()

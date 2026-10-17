"""Wire reconstruction, deterministic selection, and ANN error bounds."""
import unittest
import numpy as np
from opencood.utils import quantized_feature_packet as codec


class QuantizedPacketTest(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[0, 0, i] for i in range(7)])
        self.meta = dict(shape_zyx=[3, 5, 7], frame_id='audit', source_to_ego=np.eye(4).tolist())

    def test_variable_bits_and_padding(self):
        for bits in (2, 3, 4, 8):
            values = (np.arange(13) % (2 ** bits)).astype(np.uint8)
            np.testing.assert_array_equal(codec.unpack(codec.pack(values, bits), len(values), bits), values)
        with self.assertRaises(ValueError): codec.unpack(np.array([255], np.uint8), 1, 3)

    def test_count4_lossless_and_empty(self):
        for values in (np.zeros((7, 128)), np.random.RandomState(7).randint(0, 5, (7, 128))):
            options = dict(bits=3, keep_fraction=1.)
            packet = codec.encode(self.coords, values, self.meta, 'count4', options)
            c, v, m = codec.decode(packet, 'count4', options)
            keep = np.any(values != 0, axis=1)
            np.testing.assert_array_equal(c, self.coords[keep]); np.testing.assert_array_equal(v, values[keep])
            self.assertEqual(m, self.meta)

    def test_ann_roundtrip_and_error_bound(self):
        values = np.random.RandomState(4).exponential(size=(7, 128)).astype(np.float32)
        values[:, 0] = 0
        for bits in (2, 3, 4, 8):
            options = dict(bits=bits, keep_fraction=1.)
            c, q, scales = codec.prepare(self.coords, values, 'relu', options)
            packet = codec.encode(self.coords, values, self.meta, 'relu', options)
            decoded, f, m = codec.decode(packet, 'relu', options)
            np.testing.assert_array_equal(decoded, c)
            np.testing.assert_array_equal(f, q.astype(np.float32) * scales)
            self.assertTrue(np.all(np.abs(f - values) <= scales / 2 + 1e-6))
            self.assertEqual(m, self.meta)

    def test_selection_ties_order_and_limit(self):
        values = np.ones((7, 128), dtype=np.float32)
        options = dict(bits=3, keep_fraction=.25)
        a = codec.encode(self.coords, values, self.meta, 'count4', options)
        b = codec.encode(self.coords[::-1], values[::-1], self.meta, 'count4', options)
        self.assertEqual(a, b)
        c, _, _ = codec.decode(a, 'count4', options)
        np.testing.assert_array_equal(c, self.coords[:2])

    def test_invalid_values_and_contract(self):
        options = dict(bits=3, keep_fraction=1.)
        values = np.ones((7, 128), dtype=np.float32)
        packet = codec.encode(self.coords, values, self.meta, 'count4', options)
        with self.assertRaises(ValueError): codec.decode(packet, 'relu', options)
        for bad in (-1., np.inf, np.nan, .1, 5.):
            values[0, 0] = bad
            with self.assertRaises(ValueError): codec.encode(self.coords, values, self.meta, 'count4', options)
        with self.assertRaises(ValueError): codec.validate_options(dict(bits=3, keep_fraction=0))


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
from opencood.tools.test_e3dsnn_event_stream import metadata
from opencood.tools.profile_e3dsnn_event_granularity import encode_snapshot,decode_snapshot


class SnapshotTest(unittest.TestCase):
    def test_sparse_counts_and_empty(self):
        m=metadata();m['max_spikes']=8
        a=np.zeros((1,32)+tuple(m['contract']['shape_zyx']),dtype=np.uint8)
        self.assertEqual(encode_snapshot(a,m),b'')
        np.testing.assert_array_equal(a,decode_snapshot(b'',m))
        a.ravel()[[0,3,15,60,127]]=[1,8,2,3,4]
        np.testing.assert_array_equal(a,decode_snapshot(encode_snapshot(a,m),m))

    def test_wrong_geometry_basis_corruption_and_ceiling(self):
        m=metadata();m['max_spikes']=8
        a=np.ones((1,32)+tuple(m['contract']['shape_zyx']),dtype=np.uint8)
        packet=encode_snapshot(a,m)
        wrong=metadata();wrong['max_spikes']=8;wrong['contract']['receiver_frame']='different'
        with self.assertRaises(ValueError):decode_snapshot(packet,wrong)
        damaged=bytearray(packet);damaged[-1]^=1
        with self.assertRaises(ValueError):decode_snapshot(bytes(damaged),m)
        with self.assertRaises(ValueError):decode_snapshot(packet[:-2],m)
        a.ravel()[0]=9
        with self.assertRaises(ValueError):encode_snapshot(a,m)


if __name__=='__main__':unittest.main()

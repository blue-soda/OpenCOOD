import unittest
import numpy as np
from opencood.tools.test_e3dsnn_event_stream import metadata
from opencood.utils import event_stream_compact as compact


class CompactTest(unittest.TestCase):
    def test_content_order_metadata_and_repacking(self):
        ids=np.array([4,1,4,1,63,0,127],dtype='<u4')
        for capacity in (1,3,256):
            for compressed in (False,True):
                packets=compact.pack_all(ids,metadata(),capacity,compressed)
                restored=np.concatenate([compact.decode(p,metadata())[0] for p in packets])
                np.testing.assert_array_equal(restored,ids)
                acc=compact.CompactAccumulator(metadata())
                for p in packets:acc.receive(p)
                np.testing.assert_array_equal(acc.finish().ravel(),np.bincount(ids,minlength=128))

    def test_corruption_replay_and_unpaid_session_rejected(self):
        ids=np.array([1,2,3],dtype='<u4');m=metadata()
        packets=compact.pack_all(ids,m,2)
        acc=compact.CompactAccumulator(m)
        with self.assertRaises(ValueError):acc.receive(packets[1])
        damaged=bytearray(packets[0]);damaged[-1]^=1
        with self.assertRaises(ValueError):acc.receive(bytes(damaged))
        acc.receive(packets[0])
        with self.assertRaises(ValueError):acc.receive(packets[0])
        wrong=metadata();wrong['contract']['receiver_frame']='different'
        with self.assertRaises(ValueError):compact.decode(packets[1],wrong)
        acc.receive(packets[1]);self.assertEqual(int(acc.finish().sum()),3)

    def test_budget_against_exhaustive_single_block(self):
        ids=np.arange(80,dtype='<u4');m=metadata()
        for compressed in (False,True):
            sizes=[len(compact.encode(ids[:n],m,0,compressed)) for n in range(1,len(ids)+1)]
            for budget in (0,min(sizes)-1,min(sizes),sizes[39],max(sizes),2000):
                packets,n,used=compact.pack_prefix(ids,m,budget,256,compressed)
                expected=max([0]+[i+1 for i,s in enumerate(sizes) if s<=budget])
                self.assertEqual(n,expected);self.assertEqual(used,sum(map(len,packets)))
                self.assertLessEqual(used,budget)

    def test_multiblock_and_empty_budget(self):
        ids=np.arange(128,dtype='<u4')
        for budget in (0,1000,1500,5000):
            packets,n,used=compact.pack_prefix(ids,metadata(),budget,17)
            acc=compact.CompactAccumulator(metadata())
            for p in packets:acc.receive(p)
            self.assertEqual(int(acc.finish().sum()),n);self.assertLessEqual(used,budget)
        self.assertEqual(compact.pack_all(ids[:0],metadata()),[])


if __name__=='__main__':
    unittest.main()

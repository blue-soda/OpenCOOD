import copy
import unittest
import numpy as np
from opencood.tools.test_e3dsnn_event_stream import metadata
from opencood.utils import task_snapshot_packet as wire


def meta(family):
    m=metadata();m['contract']['channel_basis']='task32-'+'b'*64
    return dict(contract=m['contract'],source_to_ego=m['source_to_ego'],schema='task-count32-v1',family=family)


class TaskSnapshotTest(unittest.TestCase):
    def test_scaled_and_unit_roundtrip(self):
        rng=np.random.RandomState(11)
        for family in ('if8','ann2bit'):
            m=meta(family);maximum=8 if family=='if8' else 3
            scales=np.ones(32,dtype=np.float32) if family=='if8' else rng.uniform(.01,5,32).astype(np.float32)
            a=rng.randint(0,maximum+1,(1,32,1,1,4)).astype(np.uint8)
            p=wire.encode(a,scales,m);b,s=wire.decode(p,m)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(s,scales)
            wrong=copy.deepcopy(m);wrong['contract']['channel_basis']='task32-'+'c'*64
            with self.assertRaises(ValueError):wire.decode(p,wrong)
            wrong=copy.deepcopy(m);wrong['source_to_ego'][0][3]=1
            with self.assertRaises(ValueError):wire.decode(p,wrong)
            with self.assertRaises(ValueError):wire.decode(p[:-1],m)
            bad=bytearray(p);bad[-1]^=1
            with self.assertRaises(ValueError):wire.decode(bytes(bad),m)
            # Changing the sender's scale actually changes paid wire bytes.
            if family=='ann2bit':self.assertNotEqual(p,wire.encode(a,scales*2,m))

    def test_first_overflow_against_brute_force(self):
        for family in ('if8','ann2bit'):
            m=meta(family);s=np.arange(1,33,dtype=np.float32) if family=='ann2bit' else np.ones(32,dtype=np.float32)
            ids=np.array([0,5,3,0,9,6,3,8,5],dtype=np.uint32)
            full=np.bincount(ids,minlength=128).reshape(1,32,1,1,4)
            order=np.array([0,1,3,2]);candidates=[]
            for kind in ('events','rows'):
                a=np.zeros_like(full);candidates=[]
                for i in (ids if kind=='events' else order):
                    if kind=='events':a.ravel()[i]+=1
                    else:a.reshape(32,4)[:,i]=full.reshape(32,4)[:,i]
                    candidates.append((wire.encode(a,s,m),a.copy()))
                for budget in [0]+[len(p)+off for p,_ in candidates for off in (-1,0,1)]:
                    expected=b'';counts=np.zeros_like(full)
                    for p,a in candidates:
                        if len(p)>budget:break
                        expected,counts=p,a
                    kwargs=dict(addresses=ids) if kind=='events' else dict(row_order=order)
                    p,a,r=wire.select(full,s,m,budget,**kwargs)
                    self.assertEqual(p,expected);np.testing.assert_array_equal(a,counts)
                    self.assertLessEqual(len(p),budget)

    def test_invalid_scales_counts_order_and_silence(self):
        a=np.zeros((1,32,1,1,4));s=np.ones(32,dtype=np.float32);m=meta('ann2bit')
        self.assertEqual(wire.encode(a,s,m),b'')
        for scales in (s*0,s*np.nan,s.astype(np.float64)):
            with self.assertRaises(ValueError):wire.encode(a,scales,m)
        with self.assertRaises(ValueError):wire.encode(a+4,s,m)
        with self.assertRaises(ValueError):wire.select(a,s,m,2048,row_order=np.array([0]))
        with self.assertRaises(ValueError):wire.select(a,s,m,2048,addresses=np.array([1]))
        with self.assertRaises(ValueError):wire.encode(a,s*2,meta('if8'))


if __name__=='__main__':unittest.main()

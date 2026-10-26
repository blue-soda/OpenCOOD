"""Prepared CPU inputs must not be randomly voxelized again during timing."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/strong_ann_baselines'))
from compute_infocom_adapter import InfoComAdapter


class NoRandomAccess:
    def __getitem__(self,index):
        raise AssertionError('Timing must not shuffle/revoxelize the input')


class Wire:
    messages=[]


class AdapterInputTests(unittest.TestCase):
    def test_repetitions_reuse_cpu_inputs_without_message_cache(self):
        batch={'ego':{'sample_idx':579}}
        ids=['unused']*580;ids[579]='example'
        wire=Wire();adapter=InfoComAdapter(None,NoRandomAccess(),wire,ids,{579:batch})
        for _ in range(3):
            fid,got=adapter.prepare_cpu(579)
            self.assertEqual(fid,'example');self.assertIs(got,batch)
            wire.messages=[({'sender':1},b'old')]
            adapter.reset(fid);self.assertEqual(wire.messages,[])

    def test_wrong_cached_identity_is_rejected(self):
        adapter=InfoComAdapter(None,NoRandomAccess(),Wire(),['zero'],{0:{'ego':{'sample_idx':1}}})
        with self.assertRaises(ValueError):adapter.prepare_cpu(0)


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from opencood.utils import quantized_feature_packet as codec


class SenderRankTest(unittest.TestCase):
    def test_score_selection_zero_rows_ties_and_wire(self):
        coords = np.array([[0,0,x] for x in range(5)])
        values = np.ones((5,128),dtype=np.float32);values[0]=0
        scores = np.array([100.,-1.,3.,3.,2.])
        options = dict(bits=3,keep_fraction=.5)
        meta = dict(shape_zyx=[1,1,5])
        packet=codec.encode(coords,values,meta,'count4',options,scores)
        c,v,_=codec.decode(packet,'count4',options)
        np.testing.assert_array_equal(c,coords[[2,3]])
        np.testing.assert_array_equal(v,values[[2,3]])
        self.assertEqual(packet,codec.encode(coords[::-1],values[::-1],meta,'count4',options,scores[::-1]))
        one=dict(bits=3,keep_fraction=.25)
        np.testing.assert_array_equal(codec.prepare(coords,values,'count4',one,scores)[0],coords[[2]])

    def test_default_l1_and_ann_scales_after_selection(self):
        coords=np.array([[0,0,x] for x in range(3)])
        values=np.array([[1]*128,[3]*128,[2]*128],dtype=np.float32)
        options=dict(bits=2,keep_fraction=1/3)
        baseline=codec.prepare(coords,values,'relu',options)
        explicit=codec.prepare(coords,values,'relu',options,values.sum(1,dtype=np.float64))
        for a,b in zip(baseline,explicit):np.testing.assert_array_equal(a,b)
        c,q,s=codec.prepare(coords,values,'relu',options,np.array([4,0,0]))
        np.testing.assert_array_equal(c,coords[:1]);np.testing.assert_allclose(s,1/3)
        np.testing.assert_array_equal(q*s,values[:1])
        for bad in ([1,2], [1,np.nan,3]):
            with self.assertRaises(ValueError):codec.prepare(coords,values,'relu',options,bad)

    def test_six_eval_only_paths_and_deadline_runner(self):
        from pathlib import Path
        from opencood.tools.launch_e3dsnn_sender_rank import FAMILIES,POLICIES,commands
        self.assertEqual(len(FAMILIES)*len(POLICIES),6)
        for f in FAMILIES:
            for p in POLICIES:
                jobs=commands(Path('/rank'),f,p,dict(checkpoint='/fixed.pth'),'/fixed.yaml')
                self.assertEqual(len(jobs),2)
                cmd=jobs[0][0]
                self.assertEqual(cmd[cmd.index('--mode')+1],'eval')
                self.assertIn('--profile',cmd)
                self.assertIn('--packet-roundtrip',cmd)
                self.assertNotIn('--epochs',cmd)
                self.assertEqual(cmd[cmd.index('--output')+1],str(Path('/rank')/(f+'_'+p)))


if __name__=='__main__':unittest.main()

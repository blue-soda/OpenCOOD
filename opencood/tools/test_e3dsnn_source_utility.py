import unittest
import numpy as np
from opencood.models.fuse_modules.e3dsnn_source_utility import fit_probe,source_scores,select_suppressed
from opencood.tools.test_e3dsnn_task_snapshot import meta
from opencood.utils import task_snapshot_packet as wire


class SourceUtilityTest(unittest.TestCase):
    def test_fitted_source_only_and_constant_feature(self):
        x=np.column_stack((np.linspace(-2,2,100),np.ones(100)))
        y=np.tile(np.sign(x[:,:1]),(1,32));mask=np.ones_like(y,dtype=bool);mask[:,31]=False
        p=fit_probe(x,y,mask);test=np.array([[-1.,1.],[1.,1.]])
        result=source_scores(test,p)
        self.assertTrue(np.all(result[0,:31]<0) and np.all(result[1,:31]>0))
        self.assertTrue(np.all(result[:,31]==0))
        self.assertEqual(p['weights'].shape,(3,32))

    def test_suppression_preserves_suffix_and_real_budget(self):
        ids=np.array([0,1,0,2,1,0],dtype=np.uint32)
        full=np.bincount(ids,minlength=128).reshape(1,32,1,1,4)
        removed=np.zeros_like(full);removed.ravel()[0]=1
        packet,counts,record=select_suppressed(full,ids,4,removed,meta('if8'),2048)
        self.assertLessEqual(len(packet),2048);np.testing.assert_array_equal(counts,full-removed)
        silent,c,_=select_suppressed(full,ids,4,removed,meta('if8'),0)
        self.assertEqual(silent,b'');self.assertFalse(c.any())
        packet,c,_=select_suppressed(full,ids,4,np.zeros_like(full),meta('if8'),2048)
        expected,d,_=wire.select(full,np.ones(32,dtype=np.float32),meta('if8'),2048,addresses=ids)
        self.assertEqual(packet,expected);np.testing.assert_array_equal(c,d)
        removed.ravel()[3]=1
        with self.assertRaises(ValueError):select_suppressed(full,ids,4,removed,meta('if8'),2048)


if __name__=='__main__':unittest.main()

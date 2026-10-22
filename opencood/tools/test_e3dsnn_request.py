import copy
import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_request import request_codes,encode_request,decode_request,request_on_sender,request_order
from opencood.tools.test_e3dsnn_task_snapshot import meta


class RequestTest(unittest.TestCase):
    def context(self):
        m=meta('if8');m['contract'].update(shape_zyx=[3,5,6],origin_xyz=[0.,0.,0.],step_xyz=[1.,1.,1.]);return m

    def test_codec_edges_identity_and_corruption(self):
        m=self.context();u=torch.zeros(1,2,3,5,6);u[0,0,2,4,5]=.49
        c=request_codes(u);self.assertEqual(c.shape,(3,2,2));self.assertEqual(c[2,1,1],3)
        p=encode_request(c,m);np.testing.assert_array_equal(c,decode_request(p,m))
        mapped,_=request_on_sender(c,m,'cpu');self.assertEqual(float(mapped[0,0,2,4,5]),1.)
        self.assertEqual(float(mapped[0,0,0,4,5]),0.)
        wrong=copy.deepcopy(m);wrong['contract']['receiver_frame']='wrong'
        with self.assertRaises(ValueError):decode_request(p,wrong)
        bad=bytearray(p);bad[-1]^=1
        with self.assertRaises(ValueError):decode_request(bytes(bad),m)
        self.assertEqual(encode_request(np.zeros_like(c),m),b'')

    def test_inverse_geometry_not_forward_transform(self):
        m=self.context();c=np.zeros((3,2,2),dtype=np.uint8);c[1,0,1]=3
        # ego x=4 is requested. road x=3 maps to ego x=4 under +1 translation.
        m['source_to_ego'][0][3]=1
        out,cov=request_on_sender(c,m,'cpu')
        self.assertEqual(float(out[0,0,1,1,3]),1.)
        self.assertEqual(float(out[0,0,1,1,2]),0.)
        self.assertEqual(float(cov[0,0,1,1,5]),0.)
        m['source_to_ego'][0][3]=.5
        out,_=request_on_sender(c,m,'cpu');self.assertAlmostEqual(float(out[0,0,1,1,3]),.5,places=5)
        m['source_to_ego']=[[0.,-1.,0.,5.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        out,_=request_on_sender(c,m,'cpu')
        self.assertAlmostEqual(float(out[0,0,1,1,1]),1.,places=5)
        self.assertAlmostEqual(float(out[0,0,1,2,1]),0.,places=5)

    def test_channel_identity_multiset_and_tie_order(self):
        ids=np.array([0,4,1,5,0],dtype=np.uint32);times=np.array([.1,.2,.3,.4,.5])
        np.testing.assert_array_equal(request_order(ids,times,np.array([0,1,0,0])),[1,5,0,4,0])
        np.testing.assert_array_equal(np.sort(request_order(ids,times,np.ones(4))),np.sort(ids))
        np.testing.assert_array_equal(request_order(ids,times,np.zeros(4)),ids)


if __name__=='__main__':unittest.main()

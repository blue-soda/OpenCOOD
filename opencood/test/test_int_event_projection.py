import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_event_projection import sparse_deconv_events
from opencood.models.sub_modules.int_feature_memory import warp_bev


def shift(x,dy,dx):
    y=torch.zeros_like(x);h,w=x.shape[-2:]
    y[:,:,max(0,dy):min(h,h+dy),max(0,dx):min(w,w+dx)]=x[:,:,max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]
    return y


class EventProjectionTests(unittest.TestCase):
    def test_sparse_signed_delta_matches_dense(self):
        torch.manual_seed(303);conv=nn.ConvTranspose2d(4,3,2,2,bias=False);x=torch.randn(1,4,5,7)
        with torch.no_grad():
            for density in (0.,.1,1.):
                delta=torch.randn_like(x)*(torch.rand_like(x)<density);y,stats=sparse_deconv_events(conv,delta)
                self.assertTrue(torch.allclose(y,conv(delta),atol=1e-6,rtol=1e-5))
                self.assertTrue(torch.allclose(conv(x)+y,conv(x+delta),atol=1e-6,rtol=1e-5))
                self.assertEqual(stats['weighted_event_macs'],torch.count_nonzero(delta).item()*12)

    def test_integer_input_shift_commutes_with_double_output_shift(self):
        conv=nn.ConvTranspose2d(4,3,2,2,bias=False);x=torch.randn(1,4,5,7)
        with torch.no_grad():
            for dy,dx in ((1,0),(-1,2),(0,-2)):
                self.assertTrue(torch.equal(conv(shift(x,dy,dx)),shift(conv(x),2*dy,2*dx)))

    def test_fractional_warp_requires_dense_commutator_correction(self):
        torch.manual_seed(303);conv=nn.ConvTranspose2d(4,3,2,2,bias=False);x=torch.randn(1,4,5,7)
        a=np.eye(4);b=np.eye(4);b[0,3]=.37;bounds=[-4,-4,-1,4,4,1]
        with torch.no_grad():
            aligned,_=warp_bev(x,a,b,bounds);warped,_=warp_bev(conv(x),a,b,bounds)
            correction=conv(aligned)-warped;self.assertGreater(float(correction.abs().max()),.01)
            delta=torch.randn_like(x)*(torch.rand_like(x)<.1);event,_=sparse_deconv_events(conv,delta)
            corrected=warped+correction+event
            self.assertTrue(torch.allclose(corrected,conv(aligned+delta),atol=1e-6,rtol=1e-5))
            self.assertGreater(float((warped+event-conv(aligned+delta)).abs().max()),.01)

    def test_rejects_bias_or_training_semantics(self):
        conv=nn.ConvTranspose2d(4,3,2,2,bias=False);x=torch.zeros(1,4,5,7)
        with self.assertRaises(ValueError):sparse_deconv_events(conv,x)
        with torch.no_grad():
            with self.assertRaises(ValueError):sparse_deconv_events(nn.ConvTranspose2d(4,3,2,2),x)


if __name__=='__main__':unittest.main()

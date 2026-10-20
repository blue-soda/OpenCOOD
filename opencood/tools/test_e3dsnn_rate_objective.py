"""Check proxy units/gradients and prefix objective isolation."""
import unittest
import torch
from opencood.utils.e3dsnn_rate_objective import sparse_payload_proxy
from opencood.utils.e3dsnn_aux_loss import detection_loss


class RateObjectiveTests(unittest.TestCase):
    def test_payload_units_and_nonzero_gradient(self):
        x=torch.zeros(1,32,1,1,2)
        x[0,0,0,0,0]=3
        x.requires_grad_()
        loss=sparse_payload_proxy(x,2)
        self.assertAlmostEqual(float(loss),82/288,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[0,0,0,0,0]),0)

    def test_full_and_empty_payload(self):
        self.assertEqual(float(sparse_payload_proxy(torch.ones(1,32,1,1,2),2)),1.)
        self.assertEqual(float(sparse_payload_proxy(torch.zeros(1,32,1,1,2),0)),0.)
        with self.assertRaises(ValueError):
            sparse_payload_proxy(torch.ones(1,32,1,1,2),1)

    def test_weighting_and_rate_added_once(self):
        values=[torch.tensor(float(x),requires_grad=True) for x in (1,2,4)]
        prefixes=[dict(psm=v[None],rm=v[None],dm=v[None]) for v in values]
        output=dict(prefixes[-1],prefix_outputs=prefixes,prefix_loss_weights=[.6,.3,.1],
                    message_rate_loss=torch.tensor(.01),ego_output=prefixes[0])
        loss,parts=detection_loss(output,None,lambda p,y:p['psm'].sum(),weight=1.)
        self.assertAlmostEqual(float(loss),2.61,places=6)
        loss.backward()
        self.assertAlmostEqual(float(values[0].grad),1.6,places=6)
        self.assertAlmostEqual(float(values[1].grad),.3,places=6)
        self.assertAlmostEqual(float(values[2].grad),.1,places=6)
        self.assertIn('message_rate_loss',parts)
        output['prefix_loss_weights']=[1.,1.,1.]
        with self.assertRaises(ValueError):
            detection_loss(output,None,lambda p,y:p['psm'].sum())


if __name__=='__main__':
    unittest.main()

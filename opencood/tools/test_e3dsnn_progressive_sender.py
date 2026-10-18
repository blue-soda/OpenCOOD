import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive_sender import balanced_partition,ProgressiveSender,l1_partition
from opencood.utils import count4_compact_packet as codec


class SenderTest(unittest.TestCase):
    def test_l1_priority_and_ties_preserve_positions(self):
        x=torch.tensor([[[[[1.,4.,2.,4.,3.,1.]]]]])
        parts=l1_partition(x)
        self.assertTrue(torch.equal(parts[0],torch.tensor([[[[[0.,4.,0.,4.,0.,0.]]]]])))
        self.assertTrue(torch.equal(parts[1],torch.tensor([[[[[0.,0.,2.,0.,3.,0.]]]]])))
        self.assertTrue(torch.equal(sum(parts),x))

    def test_disjoint_nested_exact_and_codec(self):
        torch.manual_seed(4)
        for n in (0,1,2,3,14):
            x=torch.zeros(1,128,3,3,3);x.flatten(2)[:,:,:n]=torch.randint(1,5,(1,128,n)).float()
            score=torch.zeros(1,1,3,3,3,requires_grad=True)
            parts=balanced_partition(x,score)
            self.assertTrue(torch.equal(sum(parts),x))
            sizes=[]
            for p in parts:
                self.assertTrue(torch.equal(p,p.round()))
                active=(p[0]!=0).any(0);sizes.append(int(active.sum()))
                coords=torch.nonzero(active).numpy();features=p[0,:,active].T.detach().numpy()
                options=dict(bits=3,keep_fraction=1.)
                packet=codec.encode(coords,features,{'shape_zyx':[3,3,3]},options)
                c,v,_=codec.decode(packet,options)
                np.testing.assert_array_equal(c,coords);np.testing.assert_array_equal(v,features)
            self.assertLessEqual(max(sizes)-min(sizes),1)
            for i in range(3):
                for j in range(i):self.assertFalse(bool(((parts[i]!=0)&(parts[j]!=0)).any()))
            # Stable ties give first addresses to first prefix.
            self.assertEqual(int((parts[0][0,0]!=0).sum()),(n+2)//3)
            if n:self.assertEqual(float(parts[0].flatten(2)[0,0,0]),float(x.flatten(2)[0,0,0]))

    def test_surrogate_can_change_hard_order_without_changing_payload(self):
        x=torch.ones(1,1,1,1,6)
        scores=torch.nn.Parameter(torch.zeros_like(x));optimizer=torch.optim.SGD([scores],lr=2.)
        utility=torch.tensor([0.,0.,0.,0.,1.,2.]).reshape_as(x)
        initial=balanced_partition(x,scores)[0].detach().clone()
        for _ in range(8):
            optimizer.zero_grad();parts=balanced_partition(x,scores)
            loss=-(parts[0]*utility).sum();loss.backward()
            self.assertTrue(torch.isfinite(scores.grad).all());self.assertGreater(float(scores.grad.abs().sum()),0)
            optimizer.step()
        parts=balanced_partition(x,scores)
        self.assertGreater(float((parts[0]*utility).sum()),float((initial*utility).sum()))
        self.assertTrue(torch.equal(sum(parts),x))

    def test_sender_train_eval_identical_and_gradients(self):
        torch.manual_seed(9);m=ProgressiveSender(4,3)
        x=torch.randint(0,5,(1,4,3,5,5)).float()
        m.train();parts=m(x);m.eval();other=m(x)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(parts,other)))
        weights=torch.randn_like(x);loss=(parts[0]*weights).sum()+.5*(parts[1]*weights).sum();loss.backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(m.score[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.score[2].weight.grad.abs().sum()),0.)
        self.assertTrue(torch.equal(sum(parts),x))

    def test_invalid_values(self):
        x=torch.ones(1,1,3,3,3);s=torch.zeros_like(x)
        for invalid in (x*.1,x*5,x*float('nan')):
            with self.assertRaises(ValueError):balanced_partition(invalid,s)
        with self.assertRaises(ValueError):balanced_partition(x,s,temperature=0)
        with self.assertRaises(ValueError):balanced_partition(x,s+float('nan'))


if __name__=='__main__':unittest.main()

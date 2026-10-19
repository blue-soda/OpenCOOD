"""Causal state/geometry/learning checks for the joint path, not AP claims."""
import dataclasses
import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_joint_membrane import JointMembraneReceiver
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_content_message import CountContentBottleneck
from opencood.models.fuse_modules.e3dsnn_progressive_sender import ProgressiveSender


class JointMembraneTest(unittest.TestCase):
    def setup_state(self, retain=True, local=True, membrane=.1):
        receiver = JointMembraneReceiver(1, 1, retain, local).double()
        with torch.no_grad(): receiver.current_scale.fill_(.4)
        contract = MessageContract('car0', 'road0', 'road', 'a'*64,
            'joint-content32-'+'b'*64, (1, 1, 4), (0., 0., 0.), (1., 1., 1.), (1, 2))
        pose = torch.eye(4, dtype=torch.double)[None]; pose[0, 0, 3] = .5
        u = torch.full((1, 1, 1, 1, 4), membrane, dtype=torch.double)
        state = receiver.begin(u, contract, pose, 'a'*64, 'b'*64)
        return receiver, state, contract, pose

    def test_local_prethreshold_changes_response(self):
        outputs=[]
        for u in (.1, .4):
            r,s,c,p=self.setup_state(membrane=u)
            self.assertEqual(float(s.output.sum()),0)
            x=torch.zeros_like(s.output);x[...,1]=1
            r.receive(s,x,c,p,1,lambda a:a)
            outputs.append(s.output.detach())
        self.assertFalse(torch.equal(*outputs))

    def test_residual_changes_second_packet_response_and_conserves_state(self):
        out=[]
        for retain in (True,False):
            r,s,c,p=self.setup_state(retain=retain)
            first=torch.zeros_like(s.output);first[...,1]=1
            second=torch.zeros_like(s.output);second[...,2]=1
            r.receive(s,first,c,p,1,lambda a:a)
            self.assertEqual(float(s.output.sum()),0)
            r.receive(s,second,c,p,2,lambda a:a)
            out.append(s.output.detach())
            if retain:
                self.assertAlmostEqual(float((s.output+s.residual)[...,2]),.5)
        self.assertFalse(torch.equal(*out))

    def test_wrong_position_is_not_summed_at_same_cell(self):
        r,s,c,p=self.setup_state(membrane=0.)
        x=torch.zeros_like(s.output);x[...,1]=3
        r.receive(s,x,c,p,1,lambda a:a)
        self.assertEqual(float(s.output[...,0]),0)
        self.assertEqual(float(s.output[...,3]),0)
        self.assertEqual(float(s.output[...,1]),1)
        self.assertEqual(float(s.output[...,2]),1)

    def test_reject_frame_basis_pose_duplicate_and_closed(self):
        r,s,c,p=self.setup_state()
        x=torch.zeros_like(s.output);x[...,1]=1
        for wrong in (dataclasses.replace(c,receiver_frame='other'),
                      dataclasses.replace(c,channel_basis='other')):
            with self.assertRaises(ValueError):r.receive(s,x,wrong,p,1,lambda a:a)
        wrong=p.clone();wrong[0,0,3]+=1
        with self.assertRaises(ValueError):r.receive(s,x,c,wrong,1,lambda a:a)
        r.receive(s,x,c,p,1,lambda a:a)
        with self.assertRaises(ValueError):r.receive(s,x,c,p,1,lambda a:a)
        with self.assertRaises(ValueError):r.receive(s,x,c,p,2,lambda a:a)
        r.finish(s)
        with self.assertRaises(ValueError):r.receive(s,x*0,c,p,2,lambda a:a)

    def test_zero_message_identity_and_independent_frames(self):
        r,s,c,p=self.setup_state(membrane=1.2)
        before=s.output.clone(); residual=s.residual.clone()
        for k in (1,2,3):r.receive(s,torch.zeros_like(s.output),c,p,k,lambda a:a)
        self.assertTrue(torch.equal(before,s.output));self.assertTrue(torch.equal(residual,s.residual))
        other=r.begin(torch.full_like(s.output,.1),dataclasses.replace(c,receiver_frame='car1'),p,'a'*64,'b'*64)
        self.assertEqual(float(other.output.sum()),0)
        self.assertTrue(torch.equal(s.output,before))

    def test_signed_current_and_saturation_remain_in_residual(self):
        r,s,c,p=self.setup_state(membrane=4.4)
        x=torch.zeros_like(s.output);x[...,1]=3
        r.receive(s,x,c,p,1,lambda a:-a)
        self.assertAlmostEqual(float((s.output+s.residual)[...,1]),3.8)
        self.assertAlmostEqual(float(s.residual[...,3]),.4)

    def test_content_order_and_receiver_share_prefix_gradients(self):
        torch.manual_seed(31)
        content=CountContentBottleneck(4,2).double()
        sender=ProgressiveSender(2,4).double()
        receiver=JointMembraneReceiver(4,2).double()
        features=torch.rand(1,4,3,3,6,dtype=torch.double)*3+1
        message=content.encode(features);chunks=sender(message.counts,3)
        self.assertTrue(torch.equal(sum(chunks),message.counts))
        active=[(x!=0).any(1) for x in chunks]
        self.assertFalse(torch.any(active[0]&active[1]) or torch.any(active[0]&active[2]) or torch.any(active[1]&active[2]))
        c=MessageContract('c','r','r','a'*64,'joint-content32-'+'b'*64,(3,3,6),(0.,0.,0.),(1.,1.,1.),(1,2))
        p=torch.eye(4,dtype=torch.double)[None];p[0,0,3]=.2
        u=torch.rand(1,4,3,3,6,dtype=torch.double)*.3+.1
        state=receiver.begin(u,c,p,'a'*64,'b'*64)
        weights=torch.linspace(.1,2,6,dtype=torch.double).reshape(1,1,1,1,6)
        loss=0
        for k,x in enumerate(chunks,1):
            y=receiver.receive(state,x,c,p,k,content.decode)
            loss=loss+(y*weights).sum()/k
        loss.backward()
        for module in (content.project,content.expand,sender,receiver):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        before=[p.detach().clone() for p in content.parameters()]
        torch.optim.SGD(list(content.parameters())+list(sender.parameters())+list(receiver.parameters()),lr=.001).step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,content.parameters())))


if __name__=='__main__':unittest.main()

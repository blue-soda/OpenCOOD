import unittest
from dataclasses import replace
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import (
    MessageContract,ProgressiveMembraneFusion,split_counts,count_readout,check_transform)
from opencood.utils import count4_compact_packet as compact
from opencood.utils import quantized_feature_packet as legacy


class ProgressiveTest(unittest.TestCase):
    def test_pose_validation_with_cuda_tf32(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        previous=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            for device in devices:
                angle=.731;co=np.cos(angle);si=np.sin(angle)
                pose=torch.tensor([[[co,-si,0,2],[si,co,0,-3],[0,0,1,.2],[0,0,0,1]]],
                                  dtype=torch.float32,device=device)
                check_transform(pose)
                wrong=pose.clone();wrong[:,:3,:3]*=1.001
                with self.assertRaises(ValueError):check_transform(wrong)
        finally:torch.backends.cuda.matmul.allow_tf32=previous

    def fixture(self,channels=2,**kwargs):
        c=MessageContract('vehicle-A','road-A','infra','a'*64,'encoder-output-v1',
                          (3,5,5),(-2.,-2.,-1.),(1.,1.,1.),(1000,1010))
        return ProgressiveMembraneFusion(channels,**kwargs).double(),c,torch.eye(4,dtype=torch.double)[None]

    def start(self,m,c,t,x=None):
        if x is None:x=torch.zeros((1,m.channels)+c.shape_zyx,dtype=torch.double)
        return m.begin(x,c,t,c.encoder_sha256,c.channel_basis)

    def test_identity_channels_and_frame_lifetime(self):
        m,c,t=self.fixture();s=self.start(m,c,t)
        x=torch.zeros_like(s.output);x[0,1,1,2,2]=2
        m.receive(s,x,c,t,0)
        self.assertTrue(torch.equal(m.finish(s),x))
        with self.assertRaises(ValueError):m.receive(s,x,c,t,1)
        fresh=self.start(m,replace(c,receiver_frame='vehicle-B',sender_frame='road-B'),t)
        self.assertEqual(float(fresh.output.sum()),0.)

    def test_translation_and_roll_do_not_mix_positions_or_channels(self):
        m,c,t=self.fixture();t[0,0,3]=1
        s=self.start(m,c,t);x=torch.zeros_like(s.output);x[0,1,1,2,2]=2
        m.receive(s,x,c,t,0);expected=torch.zeros_like(x);expected[0,1,1,2,3]=2
        self.assertTrue(torch.equal(s.output,expected))
        # Separate cubic grid for a 90-degree roll, including Z movement.
        c=replace(c,shape_zyx=(5,5,5),origin_xyz=(-2.,-2.,-2.))
        t=torch.tensor([[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]],dtype=torch.double)
        s=self.start(m,c,t);x=torch.zeros_like(s.output);x[0,0,2,3,2]=2
        m.receive(s,x,c,t,0);expected=torch.zeros_like(x);expected[0,0,3,2,2]=2
        self.assertTrue(torch.equal(s.output,expected))

    def test_contract_rejection_is_atomic(self):
        m,c,t=self.fixture();s=self.start(m,c,t);x=torch.zeros_like(s.output)
        for wrong in (replace(c,receiver_frame='other'),replace(c,sender_frame='other'),
                replace(c,channel_basis='permuted'),replace(c,encoder_sha256='b'*64),
                replace(c,timestamps_us=(1000,1011)),replace(c,origin_xyz=(0.,0.,0.))):
            with self.assertRaises(ValueError):m.receive(s,x,wrong,t,0)
        wrong=t.clone();wrong[0,0,3]=1
        with self.assertRaises(ValueError):m.receive(s,x,c,wrong,0)
        with self.assertRaises(ValueError):m.receive(s,x,c,t,1)
        self.assertEqual(s.next_packet,0)
        m.receive(s,x,c,t,0)
        with self.assertRaises(ValueError):m.receive(s,x,c,t,0)
        with self.assertRaises(ValueError):m.begin(x,c,t,'b'*64,c.channel_basis)
        wrong=t.clone();wrong[0,0,0]=2
        with self.assertRaises(ValueError):self.start(m,c,wrong)

    def test_fractional_alignment_partition_equals_one_shot(self):
        torch.manual_seed(9);m,c,t=self.fixture();t[0,0,3]=.3;t[0,1,3]=-.2
        ego=torch.randn((1,2)+c.shape_zyx,dtype=torch.double)*.2
        x=torch.randint(0,5,ego.shape).double()
        a,b=self.start(m,c,t,ego),self.start(m,c,t,ego)
        parts=split_counts(x,3);self.assertTrue(torch.equal(sum(parts),x))
        for i,p in enumerate(parts):m.receive(a,p,c,t,i)
        m.receive(b,x,c,t,0)
        self.assertTrue(torch.equal(a.output,b.output))
        self.assertTrue(torch.allclose(a.residual,b.residual,atol=1e-12))

    def test_retained_fraction_triggers_and_negative_current_corrects(self):
        for retain,expected in ((True,1.),(False,0.)):
            m,c,t=self.fixture(channels=1,retain_residual=retain)
            with torch.no_grad():m.basis.weight.fill_(.4)
            ego=torch.full((1,1)+c.shape_zyx,.2,dtype=torch.double)
            s=self.start(m,c,t,ego);x=torch.ones_like(ego)
            m.receive(s,x,c,t,0);m.receive(s,x,c,t,1)
            self.assertTrue(torch.all(s.output==expected))
        m,c,t=self.fixture()
        with torch.no_grad():m.basis.weight[0,1,0,0,0]=-1
        ego=torch.zeros((1,2)+c.shape_zyx,dtype=torch.double);ego[:,0]=1
        s=self.start(m,c,t,ego);x=torch.zeros_like(ego);x[:,1]=1
        delta=m.receive(s,x,c,t,0)
        self.assertTrue(torch.all(delta[:,0]==-1));self.assertTrue(torch.all(s.output[:,0]==0))

    def test_empty_outside_and_gradients(self):
        m,c,t=self.fixture(adaptive_gate=True)
        ego=torch.randn((1,2)+c.shape_zyx,dtype=torch.double)*.1
        s=self.start(m,c,t,ego);old=s.output;res=s.residual
        m.receive(s,torch.zeros_like(ego),c,t,0)
        self.assertTrue(torch.equal(old,s.output));self.assertTrue(torch.equal(res,s.residual))
        x=torch.ones_like(ego,requires_grad=True);m.receive(s,x,c,t,1)
        s.output.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertGreater(float(m.gate.weight.grad.abs().sum()),0.)
        t[0,0,3]=100;s=self.start(m,c,t,ego);old=s.output
        m.receive(s,x.detach(),c,t,0);self.assertTrue(torch.equal(old,s.output))


class CompactTest(unittest.TestCase):
    def test_exact_nonzero_offset_including_four_and_empty(self):
        rng=np.random.RandomState(5);coords=np.array([[0,i//20,i%20] for i in range(200)])
        for density in (0.,.05,.6,1.):
            features=rng.randint(1,5,(200,128)).astype(np.float32)
            features[rng.rand(200,128)>density]=0
            meta={'shape_zyx':[3,50,126],'frame_id':'audit'}
            for fraction in (1.,.25):
                options=dict(bits=3,keep_fraction=fraction)
                old=legacy.encode(coords,features,meta,'count4',options)
                new=compact.encode(coords,features,meta,options)
                self.assertLessEqual(len(new),len(old))
                a,b,h=compact.decode(new,options);x,y,z=legacy.decode(old,'count4',options)
                np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y);self.assertEqual(h,z)
                a,b,h=compact.decode(old,options);np.testing.assert_array_equal(b,y)


if __name__=='__main__':unittest.main()

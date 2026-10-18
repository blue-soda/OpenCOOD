"""Actual content wire bytes and aligned session rejection tests."""
import unittest
from dataclasses import asdict, replace
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_content_receiver import ContentAccumulator
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
from opencood.utils import content_feature_packet as codec


class ContentPacketTest(unittest.TestCase):
    def setUp(self):
        self.basis = 'b'*64
        self.contract = MessageContract('ego1','road1','infra','a'*64,'content32-'+self.basis,
                                        (2,3,4),(0.,0.,0.),(1.,1.,1.),(101,102))
        self.pose = torch.eye(4, dtype=torch.double)[None]
        self.pose[0,0,3] = .25
        self.scales = torch.ones(1,32,1,1,1,dtype=torch.double)
        self.x = torch.zeros(1,32,2,3,4,dtype=torch.double)
        self.x[:,:,0,1,1] = torch.arange(32,dtype=torch.double) % 4

    def meta(self,k):
        return dict(contract=asdict(self.contract), basis_sha256=self.basis,
                    source_to_ego=self.pose[0].tolist(),packet_index=k,packet_count=3)

    def test_wire_all_prefixes_and_single_full_packet(self):
        for family in ('count3','ann2bit'):
            scale = np.ones(32,dtype=np.float32) if family=='count3' else np.linspace(.1,4.,32,dtype=np.float32)
            for kind in ('event','full'):
                restored = np.zeros((1,32),dtype=np.uint8)
                for k in ((1,2,3) if kind=='event' else (3,)):
                    q = (np.arange(32)%4)[None]
                    codes = (q>=k).astype(np.uint8) if kind=='event' else q
                    packet = codec.encode(np.array([[0,1,1]]),codes,scale,self.meta(k),family,kind)
                    c,v,s,m = codec.decode(packet,family,kind)
                    np.testing.assert_array_equal(c,[[0,1,1]])
                    np.testing.assert_array_equal(v,codes);np.testing.assert_array_equal(s,scale)
                    restored += v
                np.testing.assert_array_equal(restored,(np.arange(32)%4)[None])

    def test_empty_and_invalid_wire(self):
        packet = codec.encode(np.empty((0,3)),np.empty((0,32)),np.ones(32),self.meta(1),'count3')
        self.assertEqual(codec.decode(packet,'count3')[1].shape,(0,32))
        for codes in (np.full((1,32),.5),np.full((1,32),2),np.zeros((1,32)),np.ones((1,128))):
            with self.assertRaises(ValueError):codec.encode(np.array([[0,0,0]]),codes,np.ones(32),self.meta(1),'count3')
        with self.assertRaises(ValueError):codec.decode(packet,'ann2bit')
        with self.assertRaises(ValueError):codec.decode(packet,'count3','full')
        with self.assertRaises(ValueError):codec.encode(np.array([[2,0,0]]),np.ones((1,32)),np.ones(32),self.meta(1),'count3')

    def test_aligned_sum_keeps_fractional_values(self):
        receiver=ContentAccumulator(self.basis,'count3')
        state=receiver.begin(self.x,self.contract,self.pose,self.scales)
        for k in (1,2,3):
            y,_=receiver.receive(state,(self.x>=k).double(),self.contract,self.pose,self.scales,k)
            expected,_=warp_volume(self.x.clamp(max=k),self.pose,self.contract.origin_xyz,self.contract.step_xyz,geometry_dtype=torch.float64)
            torch.testing.assert_close(y,expected)
        self.assertTrue(torch.any(y != y.floor()))
        torch.testing.assert_close(receiver.finish(state),expected)
        with self.assertRaises(ValueError):receiver.receive(state,self.x,self.contract,self.pose,self.scales,1)

    def test_wrong_identity_pose_scales_or_sequence(self):
        receiver=ContentAccumulator(self.basis,'count3')
        state=receiver.begin(self.x,self.contract,self.pose,self.scales)
        event=(self.x>=1).double()
        for contract in (replace(self.contract,receiver_frame='other'),replace(self.contract,sender_frame='other'),
                         replace(self.contract,channel_basis='content32-'+'c'*64),replace(self.contract,timestamps_us=(101,999))):
            with self.assertRaises(ValueError):receiver.receive(state,event,contract,self.pose,self.scales,1)
        pose=self.pose.clone();pose[0,1,3]+=.1
        with self.assertRaises(ValueError):receiver.receive(state,event,self.contract,pose,self.scales,1)
        with self.assertRaises(ValueError):receiver.receive(state,event,self.contract,self.pose,self.scales*2,1)
        with self.assertRaises(ValueError):receiver.receive(state,event,self.contract,self.pose,self.scales,2)
        receiver.receive(state,event,self.contract,self.pose,self.scales,1)
        with self.assertRaises(ValueError):receiver.receive(state,event,self.contract,self.pose,self.scales,1)
        bad=(self.x>=2).double();bad[0,0,0,0,0]=1
        with self.assertRaises(ValueError):receiver.receive(state,bad,self.contract,self.pose,self.scales,2)

    def test_sessions_are_isolated_and_do_not_enter_state_dict(self):
        receiver=ContentAccumulator(self.basis,'ann2bit')
        a=receiver.begin(self.x,self.contract,self.pose,self.scales)
        b=receiver.begin(self.x,replace(self.contract,receiver_frame='ego2'),self.pose,self.scales)
        receiver.receive(a,(self.x>=1).double(),self.contract,self.pose,self.scales,1)
        self.assertTrue(torch.all(b.aligned==0));self.assertEqual(b.next_packet,1)
        other=ContentAccumulator(self.basis,'ann2bit')
        with self.assertRaises(ValueError):other.finish(a)


if __name__=='__main__':unittest.main()

import unittest

import torch

from opencood.models.sub_modules.int_fine_registration import OFFSETS, displace, estimate, tile_packet
from opencood.models.sub_modules.int_fine_float_receiver import FloatQuotaReceiver


class TestFineRegistration(unittest.TestCase):
    def test_float_receiver_keeps_subquantum_changes_and_ranks_signed_values(self):
        edge=FloatQuotaReceiver((0,0,-1,8,8,1),2)
        meta=dict(frame='000001',pose=torch.eye(4))
        value=torch.full((1,1,4,4),.0625)
        edge.begin(None,meta,(8,8),'initial')
        self.assertTrue(torch.equal(edge(value),value))
        state=dict(pose=meta['pose'])
        edge.commit(state,False)
        edge.begin(state,meta,(8,8),None)
        edge.active=torch.ones((1,1,2,2),dtype=torch.bool)
        target=value.clone()
        target.flatten()[:3]+=torch.tensor([.03125,-.09375,.125])
        expected=target.clone()
        expected.flatten()[0]=value.flatten()[0]
        self.assertTrue(torch.equal(edge(target),expected))
        self.assertEqual(edge.ledger['demand_addresses'],3)
        self.assertEqual(edge.ledger['bytes'],8)
        self.assertNotIn('units',edge.ledger)

    def test_partial_tiles_and_each_displacement(self):
        h,w=17,19
        generator=torch.Generator().manual_seed(303)
        old=torch.rand((1,8,h,w),generator=generator)
        levels=torch.arange(2*h*w).int().view(1,2,h,w)
        valid=torch.ones_like(old[:,:1],dtype=torch.bool)
        valid[:,:,8,8]=False
        for shift in OFFSETS:
            now=torch.roll(old,shift,(-2,-1))
            choices,support,_=estimate(old,now,valid)
            packet,decoded=tile_packet(choices,support)
            self.assertEqual(packet.numel(),5)
            self.assertTrue(torch.equal(decoded.long(),choices))
            self.assertFalse(torch.any(choices[~support]))
            got=displace(levels,decoded)
            expected=torch.where(choices!=0,torch.roll(levels,shift,(-2,-1)),levels)
            self.assertTrue(torch.equal(got,expected))
            if shift!=(0,0):
                self.assertTrue(torch.any(choices==OFFSETS.index(shift)))

    def test_shared_direction_and_opposite_placebo(self):
        old=torch.rand((1,8,24,24),generator=torch.Generator().manual_seed(304))
        valid=torch.ones_like(old[:,:1],dtype=torch.bool)
        now=torch.roll(old,(1,-1),(-2,-1))
        local=estimate(old,now,valid)[0]
        self.assertTrue(torch.equal(local,estimate(old,now,valid,'shared')[0]))
        opposite=estimate(old,now,valid,'opposite')[0]
        self.assertTrue(torch.equal(local!=0,opposite!=0))
        self.assertTrue((opposite[local!=0]==OFFSETS.index((-1,1))).all())


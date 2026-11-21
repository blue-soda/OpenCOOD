import unittest

import torch

from opencood.models.sub_modules.int_fine_registration import OFFSETS, displace, estimate, tile_packet


class TestFineRegistration(unittest.TestCase):
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


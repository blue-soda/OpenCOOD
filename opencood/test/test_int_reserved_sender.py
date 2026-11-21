"""Synthetic wire check for small changes at a guided position."""
import unittest

import torch

from opencood.models.sub_modules.int_reserved_sender import ReservedReceiver


class ReservedSenderTest(unittest.TestCase):
    def test_reserved_slots_can_deliver_small_changes(self):
        quant=torch.tensor([20]*12+[18]*12+[15]*12+[0]*36+[8]*12+[0]*12)
        value=(quant.float()*.25).reshape(1,1,1,-1)
        risk=torch.zeros_like(value)
        risk[...,72:84]=1
        read=torch.ones_like(value,dtype=torch.bool)
        for lower in (False,True):
            arm=ReservedReceiver(lower=lower)
            arm.risk,arm.frame=risk,7
            arm.refresh(value*0)
            c=arm.prepare(value,torch.arange(96),lambda x:x)
            received,stats=arm.receive(c,16,7,read,[6,12,24,48])
            self.assertEqual(stats['reserved'],2)
            self.assertEqual(stats['bytes'],134)
            self.assertEqual(stats['low_sent'],2 if lower else 0)
            self.assertEqual(int((received[...,72:84]!=0).sum()),stats['low_sent'])

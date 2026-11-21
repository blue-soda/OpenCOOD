import unittest
import torch
from opencood.models.sub_modules.int_fine_codebook import FineCodebook, pack_symbols, unpack_symbols


class FineCodebookTest(unittest.TestCase):
    def test_packing_and_payload_size(self):
        for n in range(1,34):
            symbols=torch.arange(n)%8
            data=pack_symbols(symbols)
            self.assertEqual(data.numel(),(3*n+7)//8)
            self.assertTrue(torch.equal(unpack_symbols(data,n),symbols))

    def test_signed_ties_and_saturation(self):
        changes=torch.tensor([-10,-6,-3,-1,1,3,6,10],dtype=torch.int)
        expected=torch.tensor([-8,-4,-2,-1,1,2,4,8],dtype=torch.int)
        for analog in (False,True):
            sender=FineCodebook(None,8,[1,2,4,8],analog=analog)
            old=torch.arange(8,dtype=torch.int)
            sent,ledger=sender.deliver(old,torch.arange(8),changes)
            self.assertTrue(torch.equal(sent,expected))
            self.assertTrue(torch.equal(old,torch.arange(8)+expected))
            self.assertEqual(ledger['bytes'],3)


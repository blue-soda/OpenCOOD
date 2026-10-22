import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_readout_range import paired_count_readouts
from opencood.models.fuse_modules.e3dsnn_progressive import count_readout


class ReadoutRangeTest(unittest.TestCase):
    def test_only_supported_upper_range_changes(self):
        local=torch.tensor([-.6,.4,3.49,3.5,4.,7.,7.]).reshape(1,1,1,1,7)
        current=torch.tensor([0.,.1,0.,0.,.6,1.,1.]).reshape_as(local)
        supported=torch.tensor([True,True,True,True,True,True,False]).reshape_as(local)
        capped,uncapped=paired_count_readouts(local,current,supported)
        self.assertTrue(torch.equal(capped,torch.tensor([0.,1.,3.,4.,4.,4.,4.]).reshape_as(local)))
        self.assertTrue(torch.equal(uncapped,torch.tensor([0.,1.,3.,4.,5.,8.,4.]).reshape_as(local)))
        self.assertTrue(torch.equal(uncapped[~supported],count_readout(local)[~supported]))
        self.assertTrue(torch.equal(capped[uncapped<=4],uncapped[uncapped<=4]))

    def test_zero_communication_and_channel_broadcast(self):
        local=torch.arange(-4.,12.).reshape(1,2,1,2,4)
        current=torch.ones_like(local)*100
        mask=torch.zeros(1,1,1,2,4,dtype=torch.bool)
        a,b=paired_count_readouts(local,current,mask)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a,count_readout(local)))
        mask[:,:,:,:,0]=True
        a,b=paired_count_readouts(local,current,mask)
        self.assertTrue(torch.equal(a[:,:,:,:,1:],b[:,:,:,:,1:]))
        with self.assertRaises(ValueError):paired_count_readouts(local,current,mask.float())


if __name__=='__main__':unittest.main()

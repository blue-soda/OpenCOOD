import unittest

import torch

from opencood.models.sub_modules.int_grouped_receiver import Candidates, ChargedReceiver, GroupedReceiver


class GroupedReceiverTest(unittest.TestCase):
    def test_persistent_unserved_cell_fires_and_clears(self):
        receiver = ChargedReceiver(12,side=1)
        read = torch.ones(1,1,1,2,dtype=torch.bool)
        receiver.refresh(torch.zeros_like(read,dtype=torch.float))
        ids = torch.arange(2)
        for expected in (0,1):
            receiver.aligned_debt = receiver.debt
            candidate = Candidates(torch.zeros_like(read,dtype=torch.int),ids,torch.tensor([20,12]))
            receiver.receive(candidate,1,1,read,[3,6,13,28])
            self.assertEqual(receiver.event_sites.flatten().nonzero().item(),expected)
            self.assertEqual(receiver.debt.flatten()[expected].item(),0)

    def test_groups_take_priority_over_largest_individual_change(self):
        read = torch.ones(1,1,1,5,dtype=torch.bool)
        ids = torch.arange(5)
        for floating in (False,True):
            changes = torch.tensor([20,20,30,1,3],dtype=torch.float if floating else torch.int)
            receiver = GroupedReceiver(12,side=2,floating=floating)
            for limit,chosen in [(0,[]),(1,[0,1]),(2,[0,1]),(3,[0,1,2]),(5,list(range(5)))]:
                levels = torch.zeros(1,1,1,5,dtype=changes.dtype)
                _,s = receiver.receive(Candidates(levels,ids,changes),limit,12,read,[3,6,13,28])
                active = levels.flatten().nonzero().flatten().tolist()
                self.assertEqual(len(active),limit)
                self.assertTrue(set(active).issubset(chosen))
                self.assertEqual(s['eligible'],5)
                self.assertEqual(s['addresses'],limit)


if __name__=='__main__':
    unittest.main()

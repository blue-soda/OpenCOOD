import unittest

import torch

from opencood.models.sub_modules.int_grouped_receiver import Candidates, GroupedReceiver


class GroupedReceiverTest(unittest.TestCase):
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

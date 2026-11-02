import unittest,math
import torch
from torch import nn
from opencood.models.sub_modules.int_task_membrane import TaskPotentialEvents


class CurrentCandidate(nn.Module):
    def forward(self,current,old,valid):return current


class TaskMembraneTests(unittest.TestCase):
    def test_retention_is_task_memory_and_reset_removes_it(self):
        m=TaskPotentialEvents(CurrentCandidate());v=torch.ones(1,2,2,2);sent=v.clone();valid=torch.ones(1,1,2,2,dtype=torch.bool)
        with torch.no_grad():
            for _ in range(3):v,sent,_,_=m(torch.zeros_like(v),v,sent,valid,.1)
            self.assertTrue(torch.allclose(v,torch.full_like(v,math.exp(-1.5))))
            reset,_,_,_=m(torch.zeros_like(v),v,sent,~valid,0.)
            self.assertTrue(torch.equal(reset,torch.zeros_like(v)))

    def test_stationary_state_has_no_repeated_absolute_firing(self):
        m=TaskPotentialEvents(CurrentCandidate());x=torch.full((1,2,2,2),1.1);valid=torch.zeros(1,1,2,2,dtype=torch.bool)
        with torch.no_grad():
            v,sent,_,first=m(x,torch.zeros_like(x),torch.zeros_like(x),valid,0.)
            self.assertGreater(first['signed_unit_events'],0)
            for _ in range(4):
                v,sent,_,stats=m(x,v,sent,~valid,.1);self.assertEqual(stats['signed_unit_events'],0)

    def test_signed_events_reconstruct_with_bounded_error(self):
        m=TaskPotentialEvents(CurrentCandidate());v=torch.zeros(1,3,4,5);sent=v.clone();valid=torch.zeros(1,1,4,5,dtype=torch.bool)
        positive=negative=0
        with torch.no_grad():
            for i in range(6):
                old=sent;v,sent,counts,stats=m(torch.randn_like(v)*3,v,sent,valid,.1)
                expected=torch.where(valid,old,torch.zeros_like(old))+.25*counts
                self.assertTrue(torch.equal(sent,expected));self.assertLess(stats['max_representation_error'],.25001)
                self.assertEqual(stats['signed_unit_events'],stats['positive_events']+stats['negative_events'])
                self.assertEqual(stats['state_bytes'],2*v.numel()*v.element_size());positive+=stats['positive_events'];negative+=stats['negative_events'];valid[:]=True
        self.assertGreater(positive,0);self.assertGreater(negative,0)

    def test_dense_training_and_invalid_time_are_rejected(self):
        m=TaskPotentialEvents(CurrentCandidate());x=torch.zeros(1,1,2,2);valid=torch.ones_like(x,dtype=torch.bool)
        with self.assertRaises(ValueError):m(x,x,x,valid,.1)
        with torch.no_grad():
            for dt in (-.1,.3):
                with self.assertRaises(ValueError):m(x,x,x,valid,dt)


if __name__=='__main__':unittest.main()

import unittest
import torch
from opencood.utils.e3dsnn_response_distill import response_distillation, add_response_supervision


class ResponseDistillTest(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(5)
        s={k:torch.randn(2,c,2,3,requires_grad=True) for k,c in [('psm',2),('rm',14),('dm',4)]}
        labels={k:torch.zeros(2,2,3,2) for k in ('pos_equal_one','neg_equal_one')}
        labels['pos_equal_one'][:,0,0,0]=1
        labels['neg_equal_one'][:,1,:,1]=1
        return s,labels

    def test_identical_zero_and_frozen_teacher_gradient(self):
        s,l=self.fixture();t={k:v.detach().clone().requires_grad_() for k,v in s.items()}
        loss,_=response_distillation(s,t,l,torch.ones(2));self.assertEqual(float(loss),0.)
        loss.backward();self.assertTrue(all(v.grad is None for v in t.values()))
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in s.values()))

    def test_nonzero_gradient_and_dropped_sample(self):
        s,l=self.fixture();t={k:torch.zeros_like(v,requires_grad=True) for k,v in s.items()}
        loss,_=response_distillation(s,t,l,torch.tensor([1.,0.]));self.assertGreater(float(loss),0)
        loss.backward()
        for v in s.values():
            self.assertGreater(float(v.grad[0].abs().sum()),0)
            self.assertEqual(float(v.grad[1].abs().sum()),0)
        self.assertTrue(all(v.grad is None for v in t.values()))

    def test_empty_labels_and_no_messages_are_zero(self):
        s,l=self.fixture();t={k:torch.zeros_like(v) for k,v in s.items()}
        loss,_=response_distillation(s,t,l,torch.zeros(2));self.assertEqual(float(loss),0)
        for v in l.values():v.zero_()
        loss,_=response_distillation(s,t,l,torch.ones(2));self.assertEqual(float(loss),0)
        loss.backward();self.assertTrue(all(float(v.grad.abs().sum())==0 for v in s.values()))

    def test_ignore_anchors_periodic_yaw_and_contract(self):
        s,l=self.fixture();t={k:v.detach().clone() for k,v in s.items()}
        t['psm'][:,0,0,1]+=10  # neither positive nor negative
        t['rm'][:,6,0,0]+=2*torch.pi
        loss,_=response_distillation(s,t,l,torch.ones(2));self.assertLess(abs(float(loss)),1e-10)
        with self.assertRaises(ValueError):response_distillation(s,t,l,torch.tensor([1.,.5]))
        l['neg_equal_one'][:,0,0,0]=1
        with self.assertRaises(ValueError):response_distillation(s,t,l,torch.ones(2))

    def test_control_preserves_loss_graph_and_teacher_runs(self):
        s,l=self.fixture()
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__();self.calls=0
            def forward(self,data):
                self.calls+=1
                return {k:torch.zeros_like(s[k]) for k in ('psm','rm','dm')}
        teacher=Teacher().eval();s['message_kept']=torch.tensor([1.,0.])
        original=s['psm'].square().sum();parts={}
        loss=add_response_supervision(original,parts,s,l,teacher,{},0.)
        self.assertIs(loss,original);self.assertEqual(teacher.calls,1)
        expected=torch.autograd.grad(original,s['psm'],retain_graph=True)[0]
        actual=torch.autograd.grad(loss,s['psm'],retain_graph=True)[0]
        self.assertTrue(torch.equal(expected,actual))
        augmented=add_response_supervision(original,parts,s,l,teacher,{},1.)
        self.assertGreater(float(augmented),float(original))
        teacher.train()
        with self.assertRaises(ValueError):add_response_supervision(original,parts,s,l,teacher,{},1.)


if __name__=='__main__':unittest.main()

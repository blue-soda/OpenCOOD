import unittest
import torch
from opencood.models.sub_modules.int_two_scale import StateUpdate
from opencood.models.sub_modules.int_mechanism_probes import SingleGateStateUpdate,FastReadoutUpdate,content_request,DeltaReadoutProbe


class MechanismTests(unittest.TestCase):
    def test_gate_initialization_and_gradients(self):
        torch.manual_seed(303);base=StateUpdate(8);gate=SingleGateStateUpdate(base,8,4)
        x=torch.randn(1,8,9,11);old=torch.rand_like(x);valid=torch.ones(1,1,9,11,dtype=torch.bool)
        candidate=base(x,old,valid)
        self.assertTrue(torch.allclose(gate(x,old,valid),(candidate+old)*.5,atol=1e-6))
        self.assertTrue(torch.equal(gate(x,old,~valid),base(x,torch.zeros_like(old),~valid)))
        gate(x,old,valid).square().mean().backward()
        self.assertGreater(float(gate.gate[-1].weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):gate(x,old,valid,dt=.3)

    def test_readout_identity_and_holes(self):
        model=FastReadoutUpdate(12,8,4);x=torch.randn(1,12,9,11);old=torch.rand(1,8,9,11)
        valid=torch.ones(1,1,9,11,dtype=torch.bool)
        self.assertTrue(torch.equal(model(x,old,valid),old))
        self.assertTrue(torch.equal(model(x,old,~valid),torch.relu(model.current(x))))

    def test_content_budget_and_holes(self):
        old=torch.ones(1,4,10,10);current=old.clone();current[...,4:6,4:6]=5
        valid=torch.ones(1,1,10,10,dtype=torch.bool);valid[...,0,:]=False
        request,_=content_request(current,old,valid,.1)
        self.assertEqual(int(request.sum()),19)
        self.assertTrue(bool(request[...,4:6,4:6].all()))
        self.assertTrue(bool(request[...,0,:].all()))

    def test_delta_border_and_repeated_updates(self):
        torch.manual_seed(303);model=DeltaReadoutProbe(5,7,3).double()
        x=torch.randn(1,5,13,17,dtype=torch.float64);state=None
        with torch.no_grad():
            for y,z in ((0,0),(12,16),(6,8),(0,16),(12,0)):
                x=x.clone();x[:,:,y,z]+=.3
                pred,state,stats=model.step(x,state)
                self.assertTrue(torch.allclose(pred,model.dense(x),atol=1e-10,rtol=1e-10))
            pred,_,stats=model.step(x,state)
            self.assertEqual(stats['executed_macs'],0)
            self.assertTrue(torch.equal(pred,state['a2']))


if __name__=='__main__':unittest.main()

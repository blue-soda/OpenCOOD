import unittest
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_readout_event_objective import event_objectives,EventObjectiveReadout
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout


class EventObjectiveTests(unittest.TestCase):
    def test_border_fanout_signed_amplitude_and_detached_reference(self):
        x=torch.zeros(1,1,3,3,dtype=torch.float64);x[0,0,0,0]=2;x[0,0,1,1]=-3;x.requires_grad_()
        fanout=F.conv2d(torch.ones_like(x),torch.ones_like(x),padding=1)
        weighted,units=event_objectives(x,torch.zeros_like(x,dtype=torch.int32),fanout,3,False)
        self.assertEqual(float(weighted),39);self.assertEqual(float(units),105)
        units.backward();self.assertEqual(float(x.grad[0,0,0,0]),12);self.assertEqual(float(x.grad[0,0,1,1]),-27)

    def test_dense_qat_predictions_and_actual_sparse_event_costs(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval();q=[[.125]*2,[.125]*3]
        observed=EventObjectiveReadout(source,q);actual=IntegerReadout(source,q,True)
        with torch.no_grad():
            for i in range(5):
                x=torch.randn(1,2,4,5);refresh=i in (0,3);observed.begin_frame(refresh);actual.refresh=refresh
                self.assertTrue(torch.equal(observed(x),actual(x)))
                self.assertEqual(int(observed.event_weighted),sum(z['actual_integer_macs'] for z in actual.statistics['layers']))
                self.assertEqual(int(observed.event_units),sum(z['unit_spike_ac'] for z in actual.statistics['layers']))
                actual.previous=actual.next_state;actual.previous_accumulators=actual.next_accumulators
                self.assertEqual(observed.reference_bytes(),(2+3)*4*5*4)

    def test_refresh_zero_and_ordinary_surrogate_gradient(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(1,2,1)).eval();m=EventObjectiveReadout(source,[[.125]])
        for p in m.parameters():p.requires_grad_(False)
        x=torch.ones(1,1,2,2,requires_grad=True);m.begin_frame(True);m(x);self.assertEqual(float(m.event_weighted),0);m.event_weighted.backward();self.assertEqual(float(x.grad.abs().sum()),0)
        y=torch.full((1,1,2,2),1.125,requires_grad=True);m.begin_frame(False);m(y);self.assertEqual(float(m.event_weighted),8);m.event_weighted.backward();self.assertTrue(bool(torch.isfinite(y.grad).all()));self.assertGreater(float(y.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_missing_frame_or_reference_rejected(self):
        m=EventObjectiveReadout(nn.Sequential(nn.Conv2d(1,1,1)),[[.1]])
        with self.assertRaises(ValueError):m(torch.zeros(1,1,2,2))
        with self.assertRaises(ValueError):event_objectives(torch.zeros(1,1,2,2),None,torch.ones(1,1,2,2),1,False)
        with self.assertRaises(ValueError):event_objectives(torch.zeros(1,1,2,2),torch.zeros(1,1,3,3,dtype=torch.int32),torch.ones(1,1,2,2),1,False)

if __name__=='__main__':unittest.main()

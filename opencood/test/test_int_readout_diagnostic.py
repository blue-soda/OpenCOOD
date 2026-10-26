import unittest
import numpy as np
import torch
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.models.sub_modules.int_short_readout import ShortReadoutState
from opencood.models.sub_modules.int_readout_diagnostic import DiagnosticReadoutState,readout_error


class TestReadoutDiagnostic(unittest.TestCase):
    @torch.no_grad()
    def test_teacher_history_does_not_enter_real_recurrence(self):
        torch.manual_seed(303);readout=Readout().eval()
        repair=LightRepair(24,4,current_channels=16).eval();repair.net[-1].bias.fill_(1.)
        args=(readout,repair,[-6,-5,-3,6,5,3])
        measured=DiagnosticReadoutState(*args);plain=ShortReadoutState(*args)
        x=torch.randn(1,16,10,12);state=other_state=None
        for i in range(4):
            meta=dict(segment='a',timestamp_us=1+i*50000,pose=np.eye(4))
            previous=None if state is None else state['value'].clone()
            value,new,_=measured.step(x,state,meta)
            other,other_state,_=plain.step(x,other_state,meta)
            self.assertTrue(torch.equal(value,other))
            if state is not None:self.assertTrue(torch.equal(state['value'],previous))
            state=new
            if i in (1,2):
                o=measured.observation
                self.assertEqual(o['depth'],i)
                self.assertTrue(torch.allclose(o['teacher_history'],o['target']+1,atol=1e-6))
                self.assertTrue(torch.allclose(o['recursive'],o['target']+i,atol=1e-6))
                self.assertTrue(torch.equal(o['teacher_warp_only'],o['target']))
            else:self.assertIsNone(measured.observation)
        measured.step(x,state,dict(segment='new',timestamp_us=1,pose=np.eye(4)))
        self.assertIsNone(measured.observation)

    def test_masked_error_uses_only_selected_cells(self):
        target=torch.tensor([[[[2.,100.]]]])
        value=torch.tensor([[[[3.,0.]]]])
        metric=readout_error(value,target,torch.tensor([[[[True,False]]]]))
        self.assertEqual(metric,dict(squared_error=1.,target_energy=4.,elements=1.))
        self.assertEqual(readout_error(value,target,torch.zeros_like(target))['elements'],0.)


if __name__=='__main__':unittest.main()

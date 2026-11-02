import unittest
import numpy as np
import torch
from torch import nn
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector
from opencood.models.sub_modules.int_readout_events import readout_leaves
from opencood.models.sub_modules.int_joint_membrane_readout import JointMembraneReadoutDetector


class JointMembraneTests(unittest.TestCase):
    def make(self):
        source=SubthresholdMembraneDetector(ProtectedMultiscaleRefreshDetector(fixture()),'carry')
        quanta=[[.25]*m.in_channels for m in readout_leaves(source.core.readout) if isinstance(m,nn.Conv2d)]
        return source,JointMembraneReadoutDetector(source,quanta,False),JointMembraneReadoutDetector(source,quanta,True)

    def test_independent_task_and_execution_states(self):
        source,dense,sparse=self.make();states=[None]*3
        with torch.no_grad():
            for i in range(5):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.21;meta=dict(timestamp_us=i*100000,segment='a',pose=pose);preds=[]
                for j,m in enumerate((source,dense,sparse)):
                    p,states[j],info=m.step_spatial(x,states[j],meta);preds.append(p)
                    expected=sum(states[j][k].numel()*states[j][k].element_size() for k in source.state_keys)
                    if j:expected+=sum(v.numel()*v.element_size() for k in ('readout','readout_accumulators') for v in states[j][k])
                    self.assertEqual(info['cache_bytes'],expected)
                for k in preds[1]:self.assertTrue(torch.equal(preds[1][k],preds[2][k]))
                for k in source.state_keys:
                    self.assertTrue(torch.equal(states[0][k],states[1][k]));self.assertTrue(torch.equal(states[0][k],states[2][k]))
                for k in ('readout','readout_accumulators'):
                    for a,b in zip(states[1][k],states[2][k]):self.assertTrue(torch.equal(a,b.to(a.dtype)))

    def test_scene_reset_clears_all_state_and_grad_guard(self):
        _,_,m=self.make();x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with self.assertRaises(ValueError):m.step_spatial(x,None,meta)
        with torch.no_grad():
            _,state,_=m.step_spatial(x,None,meta);later=dict(meta,timestamp_us=100000,segment='b')
            p,a,_=m.step_spatial(x,state,later);q,b,_=m.step_spatial(x,None,later)
            for k in p:self.assertTrue(torch.equal(p[k],q[k]))
            for k in m.state_keys:self.assertTrue(torch.equal(a[k],b[k]))
            for k in ('readout','readout_accumulators'):
                for u,v in zip(a[k],b[k]):self.assertTrue(torch.equal(u,v))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_task_membrane_detector import TaskMembraneDetector


class TaskMembraneDetectorTests(unittest.TestCase):
    def test_continuous_control_exact_and_phase_events_match_dense(self):
        base=fixture();models=[TemporalPyramidDetector(base,'leaky')]+[TaskMembraneDetector(base,m) for m in ('continuous','events_dense','events_phase')];states=[None]*4
        calls=[];handle=models[3].core.deblocks[1][0].register_forward_hook(lambda *_:calls.append(1))
        with torch.no_grad():
            for i in range(6):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.37;pose[1,3]=i*.23
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose);preds=[]
                for j,m in enumerate(models):
                    pred,states[j],info=m.step_spatial(x,states[j],meta);preds.append(pred)
                    if j>0:self.assertEqual(info['cache_bytes'],sum(v.numel()*v.element_size() for v in states[j].values() if torch.is_tensor(v)))
                for k in preds[0]:self.assertTrue(torch.equal(preds[0][k],preds[1][k]));self.assertTrue(torch.allclose(preds[2][k],preds[3][k],atol=2e-5,rtol=1e-4))
                for k in ('value','coarse','deep','sent'):self.assertTrue(torch.equal(states[2][k],states[3][k]))
            self.assertEqual(len(calls),2)
        handle.remove()

    def test_scene_reset_discards_membrane_sent_and_projection(self):
        model=TaskMembraneDetector(fixture(),'events_phase');x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta);meta=dict(meta,timestamp_us=100000,segment='b')
            pred,new,info=model.step_spatial(x,state,meta);fresh,zero,_=model.step_spatial(x,None,meta)
            self.assertEqual(info['reset'],'scene')
            for k in pred:self.assertTrue(torch.equal(pred[k],fresh[k]))
            for k in ('coarse','sent','projection'):self.assertTrue(torch.equal(new[k],zero[k]))
            with self.assertRaises(ValueError):model.step_spatial(x,new,meta)


if __name__=='__main__':unittest.main()

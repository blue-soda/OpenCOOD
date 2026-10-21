import unittest
import numpy as np
import torch
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory
from opencood.models.sub_modules.int_projected_gru import projected_gru


class ProjectedGRUTests(unittest.TestCase):
    def test_full_rank_preserves_recurrent_outputs_boundaries_and_resets(self):
        torch.manual_seed(303)
        source=INTFeatureMemory('gru',64,[-5,-4,-2,5,4,2]).eval()
        for m in source.modules():
            if isinstance(m,torch.nn.BatchNorm2d):
                m.running_mean.normal_(0,.1);m.running_var.uniform_(.8,1.2)
        candidate,energy=projected_gru(source,128);candidate.eval();left=right=None
        for frame in range(6):
            pose=np.eye(4);pose[0,3]=.1*frame
            meta=dict(segment='first' if frame<4 else 'second',timestamp_us=100000*frame,pose=pose)
            x=torch.randn(1,64,7,11)
            with torch.no_grad():
                a,left,ia=source.step(x,left,meta);b,right,ib=candidate.step(x,right,meta)
            self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=1e-5))
            self.assertTrue(torch.allclose(left['value'],right['value'],atol=2e-6,rtol=1e-5))
            self.assertEqual(ia['reset'],ib['reset'])
        self.assertTrue(all(abs(v-1)<1e-12 for v in energy.values()))

    def test_rank64_preserves_state_and_post_fusion_interface(self):
        source=INTFeatureMemory('gru',64,[-5,-4,-2,5,4,2]).eval()
        result,energy=projected_gru(source,64)
        self.assertLess(sum(p.numel() for p in result.parameters()),sum(p.numel() for p in source.parameters()))
        for name,value in source.fusion.post_fusion_conv.state_dict().items():
            self.assertTrue(torch.equal(value,result.fusion.post_fusion_conv.state_dict()[name]))
        self.assertTrue(all(0<v<=1 for v in energy.values()))
        x=torch.randn(1,64,7,11)
        y,state,_=result.step(x,None,dict(segment='a',timestamp_us=0,pose=np.eye(4)))
        self.assertEqual(tuple(y.shape),tuple(x.shape));self.assertEqual(state['value'].shape[1],65)
        y,_,_=result.step(x,state,dict(segment='a',timestamp_us=100000,pose=np.eye(4)))
        y.square().mean().backward()
        self.assertGreater(float(result.fusion.reset_gate_conv[0].weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()

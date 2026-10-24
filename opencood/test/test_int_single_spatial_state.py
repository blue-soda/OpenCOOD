import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_dependency_spatial_state import DependencyProjectedSpatialStage
from opencood.models.sub_modules.int_single_spatial_state import SingleSpatialStateStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class SingleSpatialStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        source=ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval(),8)
        self.parent=DependencyProjectedSpatialStage(source)
        self.single=SingleSpatialStateStage(source)
        self.x=torch.rand(1,16,24,32,dtype=torch.double)

    def test_no_intermediate_history_needed_at_same_core(self):
        with torch.no_grad():
            current=self.parent.observe(self.x);dense,maps=self.parent.dense_from_stem(current)
            self.assertTrue(torch.equal(self.single(self.x),dense))
            old=[torch.randn_like(v)*10 for v in maps]
            masks=[torch.rand(1,1,24,32)<.03,torch.zeros(1,1,24,32,dtype=torch.bool),torch.ones(1,1,24,32,dtype=torch.bool)]
            for core in masks:
                a,_,macs=self.parent.selected_from_stem(current,old,core)
                for execution in ('selected','masked_dense'):
                    b,state,cost=self.single.selected_from_stem(current,[old[-1]],core,execution)
                    self.assertEqual(len(state),1)
                    self.assertTrue(torch.allclose(a,b,atol=1e-10,rtol=1e-10))
                    self.assertTrue(torch.equal(b.masked_select(~core),old[-1].masked_select(~core)))
                    if execution=='selected':self.assertEqual(cost,macs)

    def test_pose_recurrence_bytes_and_layer_accounting(self):
        stages=(self.parent,self.single,self.single)
        modes=[SpatialBasisCache(stage,[-16,-12,-2,16,12,2],threshold=1e9,dilation=0,
            confidence_threshold=.1,confidence_dilation=4,execution=execution)
            for stage,execution in zip(stages,('selected','selected','masked_dense'))]
        states=[None]*3
        with torch.no_grad():
            for frame in range(9):
                pose=np.eye(4);pose[0,3]=frame*.8
                meta=dict(pose=pose,segment='b' if frame>=7 else 'a',timestamp_us=frame*100000)
                predictions=[];details=[]
                for i,mode in enumerate(modes):
                    pred,states[i],info=mode.step(self.x+frame*.02,states[i],meta)
                    predictions.append(pred);details.append(info)
                    states[i]['confidence']=torch.zeros(1,1,24,32,dtype=torch.double)
                    states[i]['confidence'][:,:,12,16]=.2
                for value in predictions[1:]:self.assertTrue(torch.allclose(value,predictions[0],atol=1e-10,rtol=1e-10))
                self.assertEqual(details[0]['spatial_state_bytes']-details[1]['spatial_state_bytes'],(8+16+8)*24*32*8)
                self.assertEqual(len(states[1]['maps']),1)
                self.assertEqual(len(details[1]['spatial_layer_write_fractions']),1)
                self.assertEqual(details[0]['spatial_layer_update_fractions'],details[1]['spatial_layer_update_fractions'])
                self.assertEqual(details[0]['spatial_extra_linear_macs'],details[1]['spatial_extra_linear_macs'])
                self.assertEqual(details[0]['spatial_mixer_executed_fraction'],details[1]['spatial_mixer_executed_fraction'])


if __name__=='__main__':unittest.main()

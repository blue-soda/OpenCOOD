import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_single_spatial_state import SingleSpatialStateStage
from opencood.models.sub_modules.int_row_spatial_state import RowSpatialStateStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class RowSpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        source=ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval(),8)
        self.parent=SingleSpatialStateStage(source)
        self.rows=RowSpatialStateStage(source)
        self.x=torch.rand(1,16,24,32,dtype=torch.double)

    def test_core_dense_fidelity_borders_and_actual_all_projection_macs(self):
        with torch.no_grad():
            current=self.rows.observe(self.x);dense,_=self.rows.dense_from_stem(current)
            old=torch.randn_like(current)*10
            corner=torch.zeros(1,1,24,32,dtype=torch.bool);corner[:,:,:2,:2]=True
            masks=[corner,torch.rand_like(corner.float())<.03,torch.zeros_like(corner),torch.ones_like(corner)]
            for mask in masks:
                original=old.clone()
                actual,cache,cost=self.rows.selected_from_stem(current,[old],mask)
                control,_,_=self.parent.selected_from_stem(current,[old],mask)
                self.assertTrue(torch.allclose(actual,control,atol=1e-10,rtol=1e-10))
                self.assertTrue(torch.allclose(actual.masked_select(mask),dense.masked_select(mask),atol=1e-10,rtol=1e-10))
                self.assertTrue(torch.equal(actual.masked_select(~mask),old.masked_select(~mask)))
                kernels=[c for b in self.rows.blocks for c in (b.conv1,b.conv2)]
                expected=sum(n*k.mix.weight.numel() for n,k in zip(self.rows.last_layer_cells,kernels))
                expected+=sum(n*k.project.weight.numel() for n,k in zip(self.rows.last_projection_cells,kernels))
                self.assertEqual(cost,expected);self.assertEqual(len(cache),1)
                self.assertTrue(torch.equal(old,original))

    def test_independent_pose_recurrence_same_bytes_and_less_projection(self):
        modes=[SpatialBasisCache(s,[-16,-12,-2,16,12,2],threshold=1e9,dilation=0,
            confidence_threshold=.1,confidence_dilation=4) for s in (self.rows,self.parent)]
        states=[None,None]
        observed_saving=False
        with torch.no_grad():
            for frame in range(10):
                pose=np.eye(4);pose[0,3]=frame*.8
                meta=dict(pose=pose,segment='b' if frame>=7 else 'a',timestamp_us=frame*100000)
                outputs=[];infos=[]
                for i,mode in enumerate(modes):
                    out,states[i],info=mode.step(self.x+frame*.02,states[i],meta)
                    outputs.append(out);infos.append(info)
                    states[i]['confidence']=torch.zeros(1,1,24,32,dtype=torch.double)
                    states[i]['confidence'][:,:,12,16]=.2
                self.assertTrue(torch.allclose(outputs[0],outputs[1],atol=1e-10,rtol=1e-10))
                self.assertEqual(infos[0]['spatial_state_bytes'],infos[1]['spatial_state_bytes'])
                self.assertEqual(infos[0]['spatial_layer_update_fractions'],infos[1]['spatial_layer_update_fractions'])
                self.assertLessEqual(infos[0]['spatial_projection_executed_fraction'],infos[1]['spatial_projection_executed_fraction'])
                observed_saving |= infos[0]['spatial_projection_executed_fraction']<infos[1]['spatial_projection_executed_fraction']
            self.assertTrue(observed_saving)


if __name__=='__main__':unittest.main()

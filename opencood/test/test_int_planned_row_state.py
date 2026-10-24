import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_row_spatial_state import RowSpatialStateStage
from opencood.models.sub_modules.int_planned_row_state import PlannedRowSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class PlannedRowsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        source=ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval(),8)
        self.parent=RowSpatialStateStage(source)
        self.planned=PlannedRowSpatialStage(source)
        self.x=torch.rand(1,16,24,32,dtype=torch.double)

    def test_masks_borders_exact_output_and_macs(self):
        with torch.no_grad():
            current=self.parent.observe(self.x);old=torch.randn_like(current)
            border=torch.zeros(1,1,24,32,dtype=torch.bool);border[:,:,-1,:]=True;border[:,:,:,0]=True
            for mask in (border,torch.rand_like(border.float())<.03,torch.zeros_like(border),torch.ones_like(border)):
                original=old.clone()
                a,_,cost=self.parent.selected_from_stem(current,[old],mask)
                b,_,actual=self.planned.selected_from_stem(current,[old],mask)
                self.assertTrue(torch.equal(a,b));self.assertEqual(cost,actual)
                self.assertTrue(torch.equal(old,original))
                self.assertEqual(self.parent.last_layer_cells,self.planned.last_layer_cells)
                self.assertEqual(self.parent.last_projection_cells,self.planned.last_projection_cells)
        expected=sum(c.mix.weight.numel()*8 for block in self.planned.blocks for c in (block.conv1,block.conv2))
        packed=sum(v.numel()*v.element_size() for name,v in self.planned.named_buffers() if name.endswith('packed_weight'))
        self.assertEqual(packed,expected)

    def test_pose_recurrence_no_persistent_geometry_or_state_change(self):
        modes=[SpatialBasisCache(s,[-16,-12,-2,16,12,2],threshold=1e9,dilation=0,
            confidence_threshold=.1,confidence_dilation=4) for s in (self.parent,self.planned)]
        states=[None,None]
        with torch.no_grad():
            for frame in range(10):
                pose=np.eye(4);pose[0,3]=frame*.8
                meta=dict(pose=pose,segment='b' if frame>=7 else 'a',timestamp_us=frame*100000)
                outputs=[];infos=[]
                for i,mode in enumerate(modes):
                    output,states[i],info=mode.step(self.x+frame*.02,states[i],meta)
                    outputs.append(output);infos.append(info)
                    states[i]['confidence']=torch.zeros(1,1,24,32,dtype=torch.double)
                    states[i]['confidence'][:,:,12,16]=.2
                self.assertTrue(torch.equal(outputs[0],outputs[1]))
                for key in ('spatial_extra_linear_macs','spatial_state_bytes','spatial_projection_executed_fraction','spatial_layer_update_fractions'):
                    self.assertEqual(infos[0][key],infos[1][key])
                self.assertEqual(set(states[0]),set(states[1]))


if __name__=='__main__':unittest.main()

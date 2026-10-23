import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class CellAgeCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        parent = ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).eval(),8)
        self.stage = CompactProjectedSpatialStage(parent)
        self.bounds = [-16,-12,-2,16,12,2]
        self.x = torch.rand(1,16,24,32)
        self.meta = dict(segment='a',pose=np.eye(4),timestamp_us=0)

    def test_expiry_does_not_recompute_recently_updated_cells(self):
        mode = SpatialBasisCache(self.stage,self.bounds,threshold=1e9,confidence_threshold=.1,cell_age_limit_s=.25)
        state=None; counts=[]
        with torch.no_grad():
            for frame in range(4):
                if state is not None:
                    state['confidence']=torch.zeros_like(self.x[:,:1])
                    if frame==1:state['confidence'][:,:,12,16]=.2
                _,state,info=mode.step(self.x,state,dict(self.meta,timestamp_us=frame*100000))
                counts.append(info['spatial_selected_cells'])
                self.assertLess(info['spatial_max_cell_age_s'],.25)
                self.assertEqual(info['spatial_refresh'],frame==0)
            self.assertEqual(counts,[24*32,81,0,24*32-81])
            self.assertAlmostEqual(float(state['ages'][0,0,12,16]),.2,places=6)
            self.assertEqual(float(state['ages'][0,0,0,0]),0.)
            self.assertGreater(info['spatial_full_age'],.25)

    def test_pose_reset_and_own_recurrence_matches_dense_control(self):
        modes=[SpatialBasisCache(self.stage,self.bounds,.5,execution=e,cell_age_limit_s=.25) for e in ('selected','masked_dense')]
        states=[None,None]
        with torch.no_grad():
            for frame in range(9):
                pose=np.eye(4);pose[0,3]=frame*.6
                meta=dict(self.meta,pose=pose,segment='b' if frame>=7 else 'a',timestamp_us=frame*100000)
                outputs=[];infos=[]
                for i,mode in enumerate(modes):
                    out,states[i],info=mode.step(self.x,states[i],meta)
                    outputs.append(out);infos.append(info)
                self.assertTrue(torch.allclose(*outputs,atol=2e-6,rtol=1e-5))
                self.assertTrue(torch.equal(states[0]['ages'],states[1]['ages']))
                self.assertEqual(infos[0]['spatial_selected_cells'],infos[1]['spatial_selected_cells'])
                self.assertLess(infos[0]['spatial_max_cell_age_s'],.25)
                self.assertEqual(infos[0]['spatial_refresh'],frame in (0,7))
                expected=sum(v.numel()*v.element_size() for v in states[0]['maps']+[states[0]['reference'],states[0]['ages']])
                self.assertEqual(infos[0]['spatial_state_bytes'],expected)
            with self.assertRaises(ValueError):modes[0].step(self.x,states[0],meta)
            _,fresh,info=modes[0].step(self.x,states[0],meta,reset=True)
            self.assertTrue(info['spatial_refresh']);self.assertEqual(float(fresh['ages'].max()),0.)
        with self.assertRaises(ValueError):SpatialBasisCache(self.stage,self.bounds,cell_age_limit_s=1.)


if __name__=='__main__':unittest.main()

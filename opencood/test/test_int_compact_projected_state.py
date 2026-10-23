import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class CompactProjectedStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.parent = ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval(),8)
        self.stage = CompactProjectedSpatialStage(self.parent)
        self.x = torch.rand(1,16,24,32,dtype=torch.double)

    def test_dense_and_same_mask_function_and_cost(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.parent(self.x),self.stage(self.x)))
            stem = self.parent.observe(self.x)
            _,old = self.parent.dense_from_stem(stem)
            compact = []
            for i,b in enumerate(self.parent.blocks):compact.extend((b.conv2.project(old[2*i]),old[2*i+1]))
            for fraction in (0.,.4,1.):
                mask = torch.rand(1,1,24,32)<fraction
                a,maps,macs = self.stage.selected_from_stem(stem,compact,mask)
                b,_,_ = self.stage.selected_from_stem(stem,compact,mask,'masked_dense')
                c,_,_ = self.parent.selected_from_stem(stem,old,mask)
                self.assertTrue(torch.allclose(a,b,atol=1e-11,rtol=1e-11))
                self.assertTrue(torch.allclose(a,c,atol=1e-11,rtol=1e-11))
                self.assertEqual(macs,int(mask.sum())*(4*8*16*9+2*8*16))
                for saved,initial in zip(maps,compact):self.assertTrue(torch.equal(saved.masked_select(~mask),initial.masked_select(~mask)))

    def test_recursive_warp_and_smaller_state(self):
        bounds = [-16,-12,-2,16,12,2]
        modes = [SpatialBasisCache(s,bounds,threshold=1e9,execution=e) for s,e in
                 ((self.parent,'selected'),(self.stage,'selected'),(self.stage,'masked_dense'))]
        states = [None]*3
        with torch.no_grad():
            for frame in range(7):
                pose = np.eye(4); pose[0,3] = frame*.8
                meta = dict(pose=pose,segment='b' if frame>=5 else 'a',timestamp_us=frame*100000)
                predictions=[]; infos=[]
                for i,m in enumerate(modes):
                    value,states[i],info = m.step(self.x,states[i],meta)
                    predictions.append(value); infos.append(info)
                for value in predictions[1:]:self.assertTrue(torch.allclose(value,predictions[0],atol=1e-10,rtol=1e-10))
                self.assertEqual(infos[0]['spatial_state_bytes']-infos[1]['spatial_state_bytes'],2*(16-8)*24*32*8)
                self.assertEqual(infos[1]['spatial_selected_cells'],infos[2]['spatial_selected_cells'])
                if not infos[1]['spatial_refresh']:
                    self.assertAlmostEqual(infos[1]['spatial_projection_executed_fraction'],(1+infos[1]['spatial_write_fraction'])/2)


if __name__=='__main__':unittest.main()

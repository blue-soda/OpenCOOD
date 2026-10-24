import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_dependency_spatial_state import DependencyProjectedSpatialStage,dependency_masks
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class DependencyStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        parent=ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval(),8)
        self.stage=DependencyProjectedSpatialStage(parent)
        self.x=torch.rand(1,16,24,32,dtype=torch.double)

    def test_core_is_dense_current_even_with_corrupt_history(self):
        s=self.stage
        with torch.no_grad():
            current=s.observe(self.x);dense,original=s.dense_from_stem(current)
            old=[torch.randn_like(v)*10 for v in original]
            masks=[]
            for corner in (False,True):
                mask=torch.zeros(1,1,24,32,dtype=torch.bool)
                mask[:,:,0 if corner else 12,0 if corner else 16]=True;masks.append(mask)
            masks.extend((torch.zeros_like(masks[0]),torch.ones_like(masks[0]),torch.rand_like(masks[0].float())<.03))
            for mask in masks:
                a,maps,macs=s.selected_from_stem(current,old,mask)
                b,control,_=s.selected_from_stem(current,old,mask,'masked_dense')
                supports=dependency_masks(mask);counts=[int(v.sum()) for v in supports]
                self.assertTrue(torch.allclose(a,b,atol=1e-10,rtol=1e-10))
                self.assertTrue(torch.allclose(a.masked_select(mask),dense.masked_select(mask),atol=1e-10,rtol=1e-10))
                self.assertTrue(torch.equal(a.masked_select(~mask),old[-1].masked_select(~mask)))
                for i,(v,reference) in enumerate(zip(maps,control)):
                    self.assertTrue(torch.allclose(v,reference,atol=1e-10,rtol=1e-10))
                    self.assertTrue(torch.equal(v.masked_select(~supports[i]),old[i].masked_select(~supports[i])))
                self.assertEqual(macs,sum(counts)*8*16*9+(counts[0]+counts[2])*8*16)
            self.assertEqual([int(v.sum()) for v in dependency_masks(masks[0])],[49,25,9,1])

    def test_recursive_coverage_reset_and_accounting(self):
        s=self.stage
        with self.assertRaises(ValueError):SpatialBasisCache(s,[-16,-12,-2,16,12,2])
        modes=[SpatialBasisCache(s,[-16,-12,-2,16,12,2],threshold=1e9,dilation=0,execution=e) for e in ('selected','masked_dense')]
        states=[None,None]
        with torch.no_grad():
            for frame in range(8):
                pose=np.eye(4);pose[0,3]=frame*.8
                meta=dict(pose=pose,segment='b' if frame>=6 else 'a',timestamp_us=frame*100000)
                values=[];infos=[]
                for i,mode in enumerate(modes):
                    value,states[i],info=mode.step(self.x+frame*.05,states[i],meta);values.append(value);infos.append(info)
                self.assertTrue(torch.allclose(values[0],values[1],atol=1e-10,rtol=1e-10))
                fractions=infos[0]['spatial_layer_write_fractions']
                self.assertAlmostEqual(infos[0]['spatial_mixer_executed_fraction'],sum(fractions)/4)
                self.assertAlmostEqual(infos[0]['spatial_projection_executed_fraction'],(2+fractions[0]+fractions[2])/4)
                self.assertAlmostEqual(infos[0]['spatial_write_fraction'],fractions[-1])
                self.assertEqual(infos[1]['spatial_mixer_executed_fraction'],1.)
                self.assertEqual(infos[0]['spatial_state_bytes'],infos[1]['spatial_state_bytes'])

    def test_confidence_support_is_independent_of_content_support(self):
        s=self.stage;bounds=[-16,-12,-2,16,12,2]
        for radius,expected in ((None,1),(4,81)):
            mode=SpatialBasisCache(s,bounds,threshold=1e9,dilation=0,
                confidence_threshold=.1,confidence_dilation=radius)
            with torch.no_grad():
                meta=dict(pose=np.eye(4),segment='a',timestamp_us=0)
                _,state,_=mode.step(self.x,None,meta)
                state['confidence']=torch.zeros(1,1,24,32,dtype=torch.double)
                state['confidence'][:,:,12,16]=.2
                _,_,info=mode.step(self.x,state,dict(meta,timestamp_us=100000))
                self.assertEqual(info['spatial_selected_cells'],expected)
                self.assertAlmostEqual(info['spatial_priority_fraction'],expected/(24*32),places=7)
                self.assertEqual(mode.dilation,0)
                self.assertEqual(s.last_layer_cells[-1],expected)
        with self.assertRaises(ValueError):SpatialBasisCache(s,bounds,dilation=0,confidence_dilation=2)


if __name__=='__main__':unittest.main()

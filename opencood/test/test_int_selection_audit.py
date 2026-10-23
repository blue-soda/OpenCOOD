import unittest
import copy
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache
from opencood.tools.int_selection_audit import capture_step, divergence_audit


class Detector:
    def __init__(self, stage, execution):
        self.spatial = SpatialBasisCache(stage,[-16,-12,-2,16,12,2],threshold=1e9,
            confidence_threshold=.1,execution=execution)

    def step(self, ego, state, meta):
        value, new, info = self.spatial.step(ego,None if state is None else state['spatial'],meta)
        new['confidence'] = value[:,:1].sigmoid()
        return dict(psm=value),dict(spatial=new),info


class SelectionAuditTests(unittest.TestCase):
    def test_threshold_rounding_is_separate_from_same_input_fidelity(self):
        torch.manual_seed(303)
        stage = ProjectedSpatialStage(nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).eval(),8).eval()
        selected, dense = Detector(stage,'selected'),Detector(stage,'masked_dense')
        x = torch.rand(1,16,24,32)
        meta = dict(pose=np.eye(4),segment='s',timestamp_us=0)
        with torch.no_grad():
            _,state,_ = selected.step(x,None,meta)
            state['spatial']['confidence'].zero_()
            other = copy.deepcopy(state)
            state['spatial']['confidence'][0,0,12,16] = .1-1e-8
            other['spatial']['confidence'][0,0,12,16] = .1+1e-8
            snapshots = [v.clone() for v in state['spatial']['maps']]
            meta['timestamp_us'] = 100000
            original = stage.selected_from_stem
            _,_,_,left = capture_step(selected,x,state,meta)
            _,_,_,right = capture_step(dense,x,other,meta)
            audit = divergence_audit(selected,dense,x,dict(selected=state,masked_dense=other),meta,
                dict(selected=left,masked_dense=right))
            self.assertEqual(audit['own_mask_xor_cells'],81)
            self.assertEqual(audit['prior_confidence_threshold_crossings'],1)
            self.assertTrue(audit['same_input_masks_equal'])
            self.assertTrue(audit['same_input_numerically_close'])
            self.assertEqual(stage.selected_from_stem,original)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(snapshots,state['spatial']['maps'])))
            with self.assertRaises(ValueError):capture_step(selected,x,state,dict(meta,timestamp_us=0))
            self.assertEqual(stage.selected_from_stem,original)
            original_step = dense.step
            def broken_step(*args):
                pred,new,info = original_step(*args)
                return dict(psm=pred['psm']+1.),new,info
            dense.step = broken_step
            rejected = divergence_audit(selected,dense,x,dict(selected=state,masked_dense=other),meta,
                dict(selected=left,masked_dense=right))
            self.assertTrue(rejected['same_input_masks_equal'])
            self.assertFalse(rejected['same_input_numerically_close'])


if __name__ == '__main__':unittest.main()

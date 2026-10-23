import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache
from opencood.tools.int_cache_training import detach_state


class CacheTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        parent=ProjectedSpatialStage(nn.Sequential(*[BasicBlock(8,8) for _ in range(3)]).double().eval(),4)
        self.stage=CompactProjectedSpatialStage(parent)
        for block in self.stage.blocks:
            for layer in (block.conv1,block.conv2):
                for p in layer.parameters():p.requires_grad_(True)
        self.x=torch.rand(1,8,12,16,dtype=torch.double)

    def test_explicit_opt_in_and_masked_gradients(self):
        s=self.stage
        with torch.no_grad():
            current=s.observe(self.x);_,old=s.dense_from_stem(current)
        mask=torch.zeros(1,1,12,16,dtype=torch.bool);mask[:,:,3:8,4:11]=True
        with self.assertRaises(ValueError):s.selected_from_stem(current,old,mask,'masked_dense')
        s.allow_cache_grad=True
        result,maps,_=s.selected_from_stem(current,old,mask,'masked_dense')
        with torch.no_grad():reference,_,_=s.selected_from_stem(current,old,mask,'selected')
        self.assertTrue(torch.allclose(result,reference,atol=1e-11,rtol=1e-11))
        for a,b in zip(maps,old):self.assertTrue(torch.equal(a.masked_select(~mask),b.masked_select(~mask)))
        result.square().mean().backward()
        params=[p for p in s.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in params),0.)
        self.assertTrue(all(p.grad is None for p in s.parameters() if not p.requires_grad))
        with self.assertRaises(ValueError):s.selected_from_stem(current,old,mask,'selected')

    def test_own_history_detached_and_no_bn_drift(self):
        s=self.stage;s.allow_cache_grad=True
        cache=SpatialBasisCache(s,[-8,-6,-2,8,6,2],threshold=.1,execution='masked_dense',allow_cache_grad=True)
        initial={n:b.clone() for n,b in s.named_buffers()}
        state=None;optimizer=torch.optim.SGD([p for p in s.parameters() if p.requires_grad],lr=1e-4)
        for frame in range(5):
            meta=dict(segment='a',timestamp_us=frame*100000,pose=np.eye(4))
            optimizer.zero_grad()
            value,state,_=cache.step(self.x+frame*.5,state,meta)
            value.square().mean().backward();optimizer.step()
            self.assertTrue(all(not t.requires_grad for t in state['maps']+[state['reference']]))
        self.assertTrue(all(torch.equal(initial[n],b) for n,b in s.named_buffers()))
        tensor=torch.ones(1,requires_grad=True)*2
        saved=detach_state(dict(cache=dict(value=tensor),fm=[tensor],meta=('a',3)))
        self.assertFalse(saved['cache']['value'].requires_grad)
        self.assertFalse(saved['fm'][0].requires_grad)
        self.assertEqual(saved['meta'],('a',3))


if __name__=='__main__':unittest.main()

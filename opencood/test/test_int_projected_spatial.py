import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCache


class ProjectedSpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.source = nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval()
        self.x = torch.rand(1,16,12,20,dtype=torch.double)

    def test_full_rank_preserves_mature_function(self):
        stage = ProjectedSpatialStage(self.source,16)
        with torch.no_grad():
            self.assertTrue(torch.allclose(stage(self.x),self.source(self.x),atol=1e-11,rtol=1e-11))

    def test_compressed_local_execution_mac_and_static_reuse(self):
        stage = ProjectedSpatialStage(self.source,8)
        with torch.no_grad():
            current = stage.observe(self.x)
            old = [torch.rand_like(current) for _ in range(4)]
            for fraction in (0.,.4,1.):
                mask = torch.rand(1,1,12,20) < fraction
                a,maps,macs = stage.selected_from_stem(current,old,mask)
                b,_,_ = stage.selected_from_stem(current,old,mask,'masked_dense')
                self.assertTrue(torch.allclose(a,b,atol=1e-11,rtol=1e-11))
                self.assertEqual(macs,int(mask.sum())*4*8*16*9)
                for actual,cached in zip(maps,old):
                    self.assertTrue(torch.equal(actual.masked_select(~mask),cached.masked_select(~mask)))

    def test_only_factors_train_bn_fixed_and_checkpoint_roundtrip(self):
        stage = ProjectedSpatialStage(self.source,8); stage.train()
        self.assertFalse(stage.stem.training)
        self.assertTrue(all(not m.training for m in stage.modules() if isinstance(m,nn.BatchNorm2d)))
        before = {n:p.detach().clone() for n,p in stage.named_parameters()}
        running = {n:b.clone() for n,b in stage.named_buffers()}
        parameters = [p for p in stage.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters,lr=.01)
        stage(self.x).square().mean().backward(); optimizer.step()
        self.assertTrue(all(p.grad is None for p in stage.stem.parameters()))
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in stage.named_parameters() if p.requires_grad))
        self.assertTrue(all(torch.equal(before[n],p) for n,p in stage.named_parameters() if not p.requires_grad))
        self.assertTrue(all(torch.equal(running[n],b) for n,b in stage.named_buffers()))
        restored = ProjectedSpatialStage(self.source,8); restored.load_state_dict(stage.state_dict())
        stage.eval()
        with torch.no_grad(): self.assertTrue(torch.equal(stage(self.x),restored(self.x)))

    def test_causal_cache_recurrence_and_projection_cost(self):
        stage = ProjectedSpatialStage(self.source,8).float().eval()
        bounds = [-10,-6,-2,10,6,2]
        selected = SpatialBasisCache(stage,bounds,threshold=1e9,dilation=0)
        dense = SpatialBasisCache(stage,bounds,threshold=1e9,dilation=0,execution='masked_dense')
        ls = rs = None; fractions = []
        # Hooks measure actual dense projections and native spatial calls;
        # selected functional linear MACs are reported independently.
        layers = [c for b in stage.blocks for c in (b.conv1,b.conv2)]
        cost = [0]
        def hook(layer, inputs, output):
            cost[0] += output.shape[2]*output.shape[3]*layer.weight.numel()
        hooks = [m.register_forward_hook(hook) for c in layers for m in (c.project,c.mix)]
        try:
            with torch.no_grad():
                for frame in range(7):
                    pose = np.eye(4); pose[0,3] = max(0,frame-1)
                    meta = dict(segment='b' if frame>=5 else 'a',pose=pose,timestamp_us=frame*100000)
                    x = self.x.float()*(1+frame*.02)
                    cost[0] = 0
                    a,ls,ia = selected.step(x,ls,meta)
                    actual = cost[0]
                    b,rs,ib = dense.step(x,rs,meta)
                    self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=1e-5))
                    self.assertEqual(ia['spatial_selected_cells'],ib['spatial_selected_cells'])
                    cells = x.shape[2]*x.shape[3]
                    projection = cells*sum(c.project.weight.numel() for c in layers)
                    spatial = cells*sum(c.mix.weight.numel() for c in layers)
                    self.assertEqual(actual,projection+(spatial if ia['spatial_refresh'] else 0))
                    expected = 0 if ia['spatial_refresh'] else ia['spatial_selected_cells']*sum(c.mix.weight.numel() for c in layers)
                    self.assertEqual(ia['spatial_extra_linear_macs'],expected)
                    self.assertEqual(ia['spatial_projection_executed_fraction'],1.)
                    self.assertEqual(ia['spatial_basis_executed_fraction'],0.)
                    fractions.append(ia['spatial_write_fraction'])
                self.assertEqual(fractions[1],0.)  # Zero selected cells still executes cheap projection.
                self.assertTrue(0 < fractions[2] < 1.)  # New view forces some updates.
        finally:
            for h in hooks:h.remove()


if __name__ == '__main__': unittest.main()

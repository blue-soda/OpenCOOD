import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage


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


if __name__ == '__main__': unittest.main()

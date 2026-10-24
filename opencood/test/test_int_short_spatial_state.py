import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_short_spatial_state import ShortSpatialStage


class ShortSpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.source=nn.Sequential(*[BasicBlock(16,16) for _ in range(3)]).double().eval()
        self.x=torch.randn(1,16,24,32,dtype=torch.double)

    def test_both_arms_start_from_same_retained_mature_function(self):
        plain=ShortSpatialStage(self.source)
        context=ShortSpatialStage(self.source,True)
        with torch.no_grad():
            expected=self.source[1](self.source[0](self.x))
            self.assertTrue(torch.equal(plain(self.x),expected))
            self.assertTrue(torch.equal(context(self.x),expected))
            self.assertEqual(plain(self.x).shape,self.source(self.x).shape)
            self.assertFalse(torch.allclose(expected,self.source(self.x)))

    def test_train_only_update_weights_context_and_preserve_bn_stem(self):
        for use_context in (False,True):
            stage=ShortSpatialStage(self.source,use_context).train()
            before={n:v.clone() for n,v in stage.state_dict().items()}
            self.assertFalse(stage.stem.training)
            self.assertTrue(all(not m.training for m in stage.modules() if isinstance(m,nn.BatchNorm2d)))
            parameters=[p for p in stage.parameters() if p.requires_grad]
            optimizer=torch.optim.SGD(parameters,lr=.01)
            target=self.source(self.x).detach()
            for _ in range(2):
                optimizer.zero_grad();loss=(stage(self.x)-target).square().mean();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                self.assertTrue(all(p.grad is None for p in stage.parameters() if not p.requires_grad))
                optimizer.step()
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in stage.named_parameters() if p.requires_grad))
            self.assertTrue(all(torch.equal(before[n],p) for n,p in stage.named_parameters() if not p.requires_grad))
            self.assertTrue(all(torch.equal(before[n],b) for n,b in stage.named_buffers()))
            if use_context:
                self.assertGreater(float(stage.context[0].weight.grad.abs().sum()),0.)
                self.assertGreater(float(stage.context[1].weight.grad.abs().sum()),0.)

    def test_update_bn_adaptation_preserves_initial_function_and_local_stats(self):
        for use_context in (False,True):
            stage=ShortSpatialStage(self.source,use_context)
            original={n for n,p in stage.named_parameters() if p.requires_grad}
            with torch.no_grad():expected=stage(self.x).clone()
            stage.enable_update_bn_affine()
            added={n for n,p in stage.named_parameters() if p.requires_grad}-original
            self.assertEqual(added,{'block.bn1.weight','block.bn1.bias','block.bn2.weight','block.bn2.bias'})
            before={n:v.clone() for n,v in stage.state_dict().items()}
            stage.train()
            with torch.no_grad():self.assertTrue(torch.equal(stage(self.x),expected))
            self.assertTrue(all(not m.training for m in stage.modules() if isinstance(m,nn.BatchNorm2d)))
            optimizer=torch.optim.SGD([p for p in stage.parameters() if p.requires_grad],lr=.01)
            (stage(self.x)-self.source(self.x).detach()).square().mean().backward()
            for name,p in stage.named_parameters():
                if name in added:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in stage.named_parameters() if n in added))
            self.assertTrue(all(torch.equal(before[n],p) for n,p in stage.named_parameters() if not p.requires_grad))
            self.assertTrue(all(torch.equal(before[n],b) for n,b in stage.named_buffers()))


if __name__=='__main__':unittest.main()

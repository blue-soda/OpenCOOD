import unittest
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_pointwise_spatial_state import ContextualPointwiseStage


class PointwiseSpatialStateTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(303)
        down=torch.nn.Sequential(torch.nn.Conv2d(64,128,1,2,bias=False),torch.nn.BatchNorm2d(128))
        source=torch.nn.Sequential(BasicBlock(64,128,2,down),BasicBlock(128,128),BasicBlock(128,128)).eval()
        return ContextualPointwiseStage(source)

    def test_selected_has_no_spatial_dependencies_in_update(self):
        model=self.make()
        with torch.no_grad():
            model.update[-1].weight.normal_(0,.02)
            model.update[0].weight[:,384:].normal_(0,.02)
            current,evidence=model.observe(torch.randn(1,64,15,21))
            old=torch.randn_like(current);valid=torch.rand_like(current[:,:1])>.1
            dense=F.relu(current+model.update(model.pack(evidence,old,valid)))
            for kind in ('empty','full','random'):
                mask=torch.rand_like(valid.float())>.5
                if kind!='random':mask.fill_(kind=='full')
                actual=model.selected_update(current,evidence,old,valid,mask)
                self.assertTrue(torch.allclose(actual,torch.where(mask,dense,old),atol=2e-6,rtol=1e-5))
                self.assertTrue(torch.equal(actual[~mask.expand_as(actual)],old[~mask.expand_as(old)]))

    def test_training_keeps_mature_stem_frozen(self):
        model=self.make();model.train()
        self.assertFalse(model.stem.training)
        for step in range(2):
            output=model(torch.randn(1,64,8,12));loss=(output-torch.randn_like(output)).square().mean()
            model.zero_grad();loss.backward()
            self.assertTrue(all(p.grad is None for p in model.stem.parameters()))
            self.assertTrue(torch.isfinite(loss))
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:p.add_(p.grad,alpha=-.01)
        self.assertGreater(float(model.wide.weight.grad.abs().sum()),0.)

    def test_linear_layout_matches_nonzero_history_and_counts_executed_macs(self):
        model=self.make()
        with torch.no_grad():
            model.update[-1].weight.normal_(0,.02)
            model.update[0].weight[:,384:].normal_(0,.02)
            current,evidence=model.observe(torch.randn(1,64,15,21))
            old=torch.randn_like(current);valid=torch.rand_like(current[:,:1])>.1
            dense=F.relu(current+model.update(model.pack(evidence,old,valid)))
            weights=sum(m.weight.numel() for m in model.update if isinstance(m,torch.nn.Conv2d))
            for kind in ('empty','full','random'):
                mask=torch.rand_like(valid.float())>.5
                if kind!='random':mask.fill_(kind=='full')
                actual=model.linear_selected_update(current,evidence,old,valid,mask)
                self.assertTrue(torch.allclose(actual,torch.where(mask,dense,old),atol=3e-6,rtol=1e-5))
                self.assertTrue(torch.equal(actual[~mask.expand_as(actual)],old[~mask.expand_as(old)]))
                self.assertEqual(model.last_selected_linear_macs,int(mask.sum())*weights)


if __name__=='__main__':unittest.main()

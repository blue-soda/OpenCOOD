import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_cache_repair import refresh_tiles, select_updates


class CacheRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def block(self, stride, count):
        down = nn.Sequential(nn.Conv2d(3,3,1,stride=stride,bias=False),nn.BatchNorm2d(3)) if stride>1 else None
        layer = nn.Sequential(BasicBlock(3,3,stride,down), *[BasicBlock(3,3) for _ in range(count-1)]).eval()
        for m in layer.modules():
            if isinstance(m,nn.BatchNorm2d):
                m.bias.data.uniform_(.1,.5)
                m.running_mean.uniform_(-.2,.2)
        return layer

    def test_crops_match_dense_with_boundaries_and_stride_phase(self):
        for stride,count in [(1,2),(2,1),(2,3)]:
            layer=self.block(stride,count)
            x=torch.randn(1,3,67,89)
            with torch.no_grad():
                ref=layer(x);old=torch.randn_like(ref);saved=old.clone()
                mask=torch.zeros_like(ref[:,:1],dtype=torch.bool)
                mask[...,0,0]=True;mask[...,-1,-1]=True;mask[...,13,17]=True
                radius=(1+stride)+2*stride*(count-1)
                actual,written,stats=refresh_tiles(layer,x,old,mask,stride,radius,tile=4,allow_fallback=False)
                expanded=written.expand_as(actual)
                self.assertTrue(torch.allclose(actual[expanded],ref[expanded],atol=2e-6,rtol=2e-5))
                self.assertTrue(torch.equal(actual[~expanded],old[~expanded]))
                self.assertTrue(torch.equal(saved,old))
                self.assertFalse(stats['fallback'])

    def test_empty_refresh_never_calls_module(self):
        class Fail(nn.Module):
            def forward(self,x):raise AssertionError('Should skip')
        x=torch.randn(1,3,20,24);old=x.clone()
        out,written,stats=refresh_tiles(Fail().eval(),x,old,torch.zeros_like(x[:,:1],dtype=torch.bool),1,2)
        self.assertTrue(torch.equal(out,old));self.assertEqual(stats['executed_input_area_ratio'],0)

    def test_full_refresh_and_dense_fallback(self):
        layer=self.block(2,3);x=torch.randn(1,3,33,49)
        with torch.no_grad():
            ref=layer(x);mask=torch.ones_like(ref[:,:1],dtype=torch.bool)
            out,written,stats=refresh_tiles(layer,x,torch.zeros_like(ref),mask,2,11)
            self.assertTrue(torch.equal(out,ref));self.assertTrue(bool(written.all()))
            self.assertTrue(stats['fallback']);self.assertEqual(stats['executed_input_area_ratio'],1)

    def test_gate_detects_appear_disappear_and_invalid_at_zero_budget(self):
        past=torch.zeros(1,3,35,67);current=past.clone()
        past[...,2,3]=1;current[...,7,8]=1
        valid=torch.ones_like(past[:,:1],dtype=torch.bool);valid[...,-1,-1]=False
        mask=select_updates(current,past,valid,0)
        self.assertTrue(mask[...,2,3]);self.assertTrue(mask[...,7,8]);self.assertTrue(mask[...,-1,-1])
        self.assertEqual(int(mask.sum()),3)

    def test_identical_features_no_optional_updates(self):
        x=torch.ones(1,3,35,67);valid=torch.ones_like(x[:,:1],dtype=torch.bool)
        self.assertFalse(bool(select_updates(x,x,valid,.5).any()))

    def test_reject_training_and_invalid_budget(self):
        x=torch.ones(1,3,16,16);mask=torch.ones_like(x[:,:1],dtype=torch.bool)
        with self.assertRaises(ValueError):refresh_tiles(nn.Identity(),x,x,mask,1,0)
        with self.assertRaises(ValueError):select_updates(x,x,mask,1.1)


if __name__=='__main__':unittest.main()

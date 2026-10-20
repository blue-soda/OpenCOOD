import unittest
import torch
from opencood.models.sub_modules.int_anchor_residual import AnchorResidualModules, refresh_reason
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class AnchorResidualTests(unittest.TestCase):
    def test_identity_and_equal_repair_initialization(self):
        torch.manual_seed(303); direct = AnchorResidualModules(False)
        torch.manual_seed(303); fusion = AnchorResidualModules(True)
        self.assertTrue(all(torch.equal(v, fusion.repair.state_dict()[k])
                            for k,v in direct.repair.state_dict().items()))
        e = torch.randn(1,128,9,13); h = torch.randn(1,256,9,13)
        valid = torch.ones_like(h[:,:1], dtype=torch.bool)
        for m in (direct, fusion):
            repaired = m.repair(torch.cat([e,h,valid.float()],1))
            self.assertTrue(torch.equal(repaired, h))
            self.assertTrue(torch.equal(m.decode(e,repaired,valid), h))

    def test_readout_uses_current_e_and_has_exact_local_execution(self):
        m = AnchorResidualModules(True).readout.eval()
        torch.nn.init.normal_(m.net[-1].weight, std=.02)
        packed = torch.randn(1,385,17,21,requires_grad=True)
        mask = torch.zeros(1,1,17,21,dtype=torch.bool)
        mask[...,0,0] = True; mask[...,-1,-1] = True
        actual,written,_ = refresh_tiles(m,packed,packed[:,128:384],mask,1,1,4,allow_fallback=False)
        expected = m(packed)
        self.assertTrue(torch.allclose(actual[written.expand_as(actual)],
                                      expected[written.expand_as(expected)],atol=1e-5,rtol=1e-5))
        expected.square().mean().backward()
        self.assertGreater(float(packed.grad[:,:128].abs().sum()),0.)

    def test_anchor_period_age_reset_and_causality(self):
        cache = dict(stamp=1000000,frames=0)
        meta = dict(timestamp_us=1100000)
        self.assertIsNone(refresh_reason(cache,meta,False,3))
        self.assertIsNone(refresh_reason(dict(cache,frames=1),dict(timestamp_us=1200000),False,3))
        self.assertEqual(refresh_reason(dict(cache,frames=2),dict(timestamp_us=1200000),False,3),'period')
        self.assertEqual(refresh_reason(cache,dict(timestamp_us=1250000),False,3),'age')
        self.assertEqual(refresh_reason(cache,meta,True,3),'reset')
        with self.assertRaises(ValueError):
            refresh_reason(cache,dict(timestamp_us=1000000),False,3)


if __name__ == '__main__':
    unittest.main()

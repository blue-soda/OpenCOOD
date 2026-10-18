import unittest
import torch
from opencood.models.sub_modules.int_light_repair import LightRepair,repair_input_request,make_repairs
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class LightRepairTests(unittest.TestCase):
    def test_decoded_repair_uses_same_grid_evidence_and_learns_residual(self):
        torch.manual_seed(11)
        module=make_repairs('decoded')[0]
        first=torch.randn(1,128,9,13,requires_grad=True)
        old=torch.randn(1,256,9,13);valid=torch.ones(1,1,9,13,dtype=torch.bool)
        valid[...,-1,-1]=False
        x,request=repair_input_request(first,old,valid,torch.zeros_like(valid),0)
        self.assertTrue(torch.equal(x[:,:128],first))
        self.assertTrue(torch.equal(module(x),old))
        self.assertEqual(int(request.sum()),1)
        torch.nn.init.normal_(module.net[-1].weight,std=.02)
        module(x).square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()),0.)
        self.assertGreater(float(module.net[-1].weight.grad.abs().sum()),0.)

    def test_identity_init_and_actual_cropped_parity(self):
        torch.manual_seed(7)
        module=LightRepair(256).eval();x=torch.randn(1,385,25,63)
        self.assertTrue(torch.equal(module(x),x[:,128:384]))
        torch.nn.init.normal_(module.net[-1].weight,std=.02)
        mask=torch.zeros(1,1,25,63,dtype=torch.bool)
        mask[...,0,0]=True;mask[...,-1,-1]=True;mask[...,12,32]=True
        output,written,stats=refresh_tiles(module,x,x[:,128:384],mask,1,1,4,allow_fallback=False)
        expected=module(x)
        self.assertTrue(torch.allclose(output[written.expand_as(output)],expected[written.expand_as(expected)],atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.equal(output[~written.expand_as(output)],x[:,128:384][~written.expand_as(output)]))
        self.assertLess(stats['actual_write_fraction'],1.)

    def test_direct_scale_evidence_and_new_view_requests(self):
        first=torch.ones(1,128,100,252);changed=torch.zeros(1,1,100,252,dtype=torch.bool)
        for index,channels,shape in [(1,256,(50,126)),(2,512,(25,63))]:
            old=torch.randn(1,channels,*shape);valid=torch.ones(1,1,*shape,dtype=torch.bool)
            valid[...,-1,-1]=False
            x,request=repair_input_request(first,old,valid,changed,index)
            self.assertEqual(x.shape[1],128+channels+1)
            self.assertEqual(int(request.sum()),1)
            self.assertEqual(float(x[:,:128].mean()),1.)


if __name__=='__main__':unittest.main()

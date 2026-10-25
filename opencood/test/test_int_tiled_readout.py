import unittest
import torch
from opencood.test.test_int_local_readout import Readout
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.models.sub_modules.int_tiled_readout import TiledReadoutCache


class TestTiledReadout(unittest.TestCase):
    @torch.no_grad()
    def test_same_mask_exact_core_world_border_and_cost(self):
        torch.manual_seed(710)
        model=Readout().eval();bounds=[-1,-1,-1,1,1,1]
        row=LocalReadoutCache(model,bounds,.5)
        for h,w in ((1,1),(9,13),(18,35)):
            x=torch.randn(1,16,h,w);old=torch.randn(1,24,h,w)
            dense=row.dense(x)
            for tile in (8,16):
                cache=TiledReadoutCache(model,bounds,.5,tile=tile,chunk_tiles=2)
                for kind in ('empty','full','edges','island'):
                    mask=torch.zeros(1,1,h,w,dtype=torch.bool)
                    if kind=='full':mask.fill_(True)
                    if kind=='edges':mask[:,:,:,0]=True;mask[:,:,-1,:]=True
                    if kind=='island':mask[:,:,h//2,w//2]=True
                    out,macs,support=cache.selected(x,old,mask)
                    self.assertTrue(torch.allclose(out.masked_select(mask),dense.masked_select(mask),atol=2e-6,rtol=1e-5),(h,w,tile,kind))
                    self.assertTrue(torch.equal(out.masked_select(~mask),old.masked_select(~mask)))
                    self.assertEqual(support,cache.last_tiles*(tile+2)**2)
                    expected=support*(cache.first.weight.numel()+cache.project.weight.numel())+cache.last_tiles*tile*tile*cache.second.weight.numel()
                    self.assertEqual(macs,expected)
                    self.assertEqual(cache.packed_weight_bytes,0)
                    if kind=='empty':self.assertEqual(macs,0)

    @torch.no_grad()
    def test_chunk_independence_and_inputs_unchanged(self):
        torch.manual_seed(711);model=Readout().eval();bounds=[-1,-1,-1,1,1,1]
        x=torch.randn(1,16,21,27);old=torch.randn(1,24,21,27)
        mask=torch.rand(1,1,21,27)>.8;before=x.clone();saved=old.clone()
        a=TiledReadoutCache(model,bounds,.5,tile=8,chunk_tiles=1)
        b=TiledReadoutCache(model,bounds,.5,tile=8,chunk_tiles=32)
        first,ma,_=a.selected(x,old,mask);second,mb,_=b.selected(x,old,mask)
        self.assertTrue(torch.allclose(first,second,atol=2e-6,rtol=1e-5));self.assertEqual(ma,mb)
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(old,saved))


if __name__=='__main__':unittest.main()

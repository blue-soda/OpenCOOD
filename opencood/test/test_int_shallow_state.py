import unittest
import numpy as np
import torch
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_shallow_state import ShallowStateEncoder
from opencood.models.sub_modules.int_cache_repair import refresh_tiles


class ShallowStateTests(unittest.TestCase):
    def source(self):
        torch.manual_seed(303)
        down=torch.nn.Sequential(torch.nn.Conv2d(64,128,1,2,bias=False),torch.nn.BatchNorm2d(128))
        return torch.nn.Sequential(BasicBlock(64,128,2,down),BasicBlock(128,128),BasicBlock(128,128)).eval()

    def test_dense_identity_with_arbitrary_history_and_boundary(self):
        source=self.source();model=ShallowStateEncoder(source,[-5,-4,-2,5,4,2])
        with torch.no_grad():
            for shape in ((1,1),(7,11),(16,24)):
                x=torch.randn(1,64,*shape);current=model.stem(x)
                packed=torch.cat((current,torch.randn_like(current),torch.ones_like(current[:,:1])),1)
                self.assertTrue(torch.allclose(model.update(packed),source(x),atol=1e-6,rtol=1e-5))

    def test_crop_halo_preserves_trained_history_conditioning(self):
        model=ShallowStateEncoder(self.source(),[-5,-4,-2,5,4,2])
        with torch.no_grad():
            model.update.mix.weight[:,128:].normal_(0,.01)
            packed=torch.randn(1,257,21,29);old=torch.randn(1,128,21,29)
            request=torch.zeros(1,1,21,29,dtype=torch.bool)
            request[:,:,0,0]=True;request[:,:,-1,-1]=True;request[:,:,10,14]=True
            actual,mask,_=refresh_tiles(model.update,packed,old,request,1,4,tile=4,allow_fallback=False)
            expected=torch.where(mask,model.update(packed),old)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-5,rtol=1e-5))

    def test_unchanged_reference_and_scene_refresh(self):
        model=ShallowStateEncoder(self.source(),[-5,-4,-2,5,4,2],threshold=1e9)
        meta=dict(segment='a',timestamp_us=0,pose=np.eye(4))
        with torch.no_grad():
            _,old,_=model.step(torch.randn(1,64,16,24),None,meta)
            meta=dict(meta,timestamp_us=100000)
            _,new,info=model.step(torch.randn(1,64,16,24),old,meta)
            self.assertEqual(info['actual_write_fraction'],0.)
            self.assertTrue(torch.equal(old['value'],new['value']))
            self.assertTrue(torch.equal(old['reference'],new['reference']))
            _,_,info=model.step(torch.randn(1,64,16,24),new,dict(meta,segment='b',timestamp_us=200000))
            self.assertTrue(info['refresh'])


if __name__=='__main__':unittest.main()

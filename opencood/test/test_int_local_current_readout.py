import unittest
import numpy as np
import torch
from torch.nn import functional as F
from test_int_local_readout import Readout
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_local_current_readout import LocalCurrentReadoutCache
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_short_readout import ShortReadoutState


class TestLocalCurrentReadout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.source=Readout().eval()
        self.repair,_=current_readout_from_mature(self.source,8,False)
        self.bounds=[-16.,-12.,-3.,16.,12.,3.]
        self.cache=LocalCurrentReadoutCache(self.source,self.repair,self.bounds,.5)

    @torch.no_grad()
    def test_five_frame_dense_and_local_refresh_bounds(self):
        x=torch.randn(1,16,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        caches=[LocalCurrentReadoutCache(self.source,self.repair,self.bounds,.5,period=5),
                ShortReadoutState(self.source,self.repair,self.bounds,5)]
        for cache in caches:
            _,initial,_=cache.step(x,None,meta);initial['confidence']=torch.zeros_like(x[:,:1]);state=initial
            for i in range(1,6):
                _,new,info=cache.step(x,state,dict(meta,timestamp_us=100000+i*50000))
                self.assertEqual(info['readout_refresh'],i==5)
                new['confidence']=initial['confidence'];state=new
            _,_,info=cache.step(x,initial,dict(meta,timestamp_us=550000))
            self.assertEqual(info['readout_refresh_reason'],'age')
            _,_,info=cache.step(x,initial,dict(meta,segment='new',timestamp_us=150000))
            self.assertEqual(info['readout_refresh_reason'],'reset')
            with self.assertRaises(ValueError):cache.step(x,initial,meta)
        self.assertEqual(self.cache.max_full_age_us,250000)

    @torch.no_grad()
    def test_selected_boundaries_halo_and_measured_macs(self):
        current=torch.randn(1,16,24,32);old=torch.randn(1,24,24,32);saved=old.clone()
        dense=self.repair.net(current)
        for pattern in ('none','all','corner','island','scattered'):
            mask=torch.zeros(1,1,24,32,dtype=torch.bool)
            if pattern=='all':mask.fill_(True)
            if pattern=='corner':mask[:,:,:2,:3]=True;mask[:,:,-1,-1]=True
            if pattern=='island':mask[:,:,8:10,10:12]=True
            if pattern=='scattered':mask[:,:,::5,::7]=True
            calls=[];handles=[]
            for module in self.repair.modules():
                if isinstance(module,torch.nn.Conv2d):
                    handles.append(module.register_forward_hook(lambda m,i,o:calls.append(o.shape[-2]*o.shape[-1]*m.weight.numel())))
            try:actual,macs,extra,support=self.cache.selected(current,old,mask)
            finally:
                for h in handles:h.remove()
            self.assertTrue(torch.allclose(actual,torch.where(mask,dense,old),atol=2e-5,rtol=2e-5),pattern)
            self.assertTrue(torch.equal(actual.masked_select(~mask),old.masked_select(~mask)))
            self.assertTrue(torch.equal(old,saved))
            self.assertEqual(support,int((F.max_pool2d(mask.float(),3,1,1)>0).sum()))
            expected=support*self.cache.first.weight.numel()+int(mask.sum())*(self.cache.second.weight.numel()+self.cache.output.weight.numel())
            self.assertEqual(macs,expected)
            self.assertEqual(sum(calls)+extra,macs)
            small=LocalCurrentReadoutCache(self.source,self.repair,self.bounds,.5,row_chunk_size=7)
            other,other_macs,_,other_support=small.selected(current,old,mask)
            self.assertTrue(torch.allclose(actual,other,atol=2e-5,rtol=2e-5))
            self.assertEqual((macs,support),(other_macs,other_support))
            cached=LocalCurrentReadoutCache(self.source,self.repair,self.bounds,.5,cache_geometry=True)
            other,other_macs,_,other_support=cached.selected(current,old,mask)
            self.assertTrue(torch.equal(actual,other))
            self.assertEqual((macs,support),(other_macs,other_support))

    @torch.no_grad()
    def test_complete_state_alignment_masks_refresh_and_own_history(self):
        current=torch.randn(1,16,24,32)
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        output,state,info=self.cache.step(current,None,meta)
        self.assertTrue(torch.equal(output,self.source.tail(self.source.spatial(current))))
        state['confidence']=torch.zeros(1,1,24,32)
        moved=np.eye(4);moved[0,3]=1.
        packed,valid=warp_bev(torch.cat((state['value'],state['reference'],state['confidence']),1),state['pose'],moved,self.bounds)
        next_current=packed[:,24:-1].clone();next_current[:,:,10:12,12:14]+=3.
        next_meta=dict(meta,timestamp_us=150000,pose=moved)
        output,new,info=self.cache.step(next_current,state,next_meta)
        mask=self.cache.last_mask.clone()
        dense=LocalCurrentReadoutCache(self.source,self.repair,self.bounds,.5,backend='masked_dense')
        expected,other,_=dense.step(next_current,state,next_meta)
        self.assertTrue(torch.equal(mask,dense.last_mask))
        self.assertTrue(bool((mask|valid).all()))
        self.assertTrue(bool(mask.any()) and not bool(mask.all()))
        self.assertTrue(torch.allclose(output,expected,atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.equal(output.masked_select(~mask),packed[:,:24].masked_select(~mask)))
        self.assertTrue(torch.equal(new['reference'],torch.where(mask,next_current,packed[:,24:-1])))
        self.assertTrue(torch.equal(new['value'],output))
        new['confidence']=state['confidence']
        _,new,_=self.cache.step(next_current,new,dict(next_meta,timestamp_us=200000))
        new['confidence']=state['confidence']
        _,_,info=self.cache.step(next_current,new,dict(next_meta,timestamp_us=250000))
        self.assertEqual(info['readout_refresh_reason'],'period')
        with self.assertRaises(ValueError):self.cache.step(next_current,new,dict(next_meta,timestamp_us=200000))
        _,_,info=self.cache.step(next_current,state,dict(next_meta,timestamp_us=400000))
        self.assertEqual(info['readout_refresh_reason'],'age')


if __name__=='__main__':unittest.main()

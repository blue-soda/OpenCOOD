import unittest
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.models.sub_modules.int_budgeted_readout import BudgetedReadoutCache, executed_macs


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Conv2d(16,24,3,padding=1)
        self.tail=nn.Sequential(nn.BatchNorm2d(24),nn.ReLU(),nn.Sequential(nn.Conv2d(24,8,1,bias=False),
            nn.Conv2d(8,24,3,padding=1)),nn.BatchNorm2d(24),nn.ReLU())
        for bn in (self.tail[0],self.tail[3]):
            bn.running_mean.copy_(torch.randn(24))
            bn.running_var.copy_(torch.rand(24)+.2)


class TestLocalReadout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        self.model=Readout().eval()
        self.bounds=[-6.,-5.,-3.,6.,5.,3.]
        self.cache=LocalReadoutCache(self.model,self.bounds,.25)

    @torch.no_grad()
    def test_absolute_budget_selection_and_dense_fidelity(self):
        x=torch.randn(1,16,10,12);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        for scale in (.8,1.):
            local=BudgetedReadoutCache(self.model,self.bounds,.5,budget_scale=scale)
            dense=BudgetedReadoutCache(self.model,self.bounds,.5,execution='masked_dense',budget_scale=scale)
            _,state,_=local.step(x,None,meta);state['confidence']=torch.zeros(1,1,10,12)
            y=x.clone();y[:,:,1:4,2:5]+=1
            output,new,info=local.step(y,state,dict(meta,timestamp_us=150000))
            control,other,_=dense.step(y,state,dict(meta,timestamp_us=150000))
            self.assertTrue(torch.allclose(output,control,atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.equal(local.last_mask,dense.last_mask))
            self.assertLessEqual(info['readout_selector_macs'],info['readout_selector_budget'])
            self.assertEqual(info['readout_selector_macs'],info['readout_extra_linear_macs'])
            state['confidence'].fill_(.9)
            _,_,info=local.step(y,state,dict(meta,timestamp_us=150000))
            self.assertTrue(bool(local.last_mask.all()))
            self.assertEqual(info['readout_budget_protection_override'],scale<1.)

    @torch.no_grad()
    def test_selected_complete_dependencies_and_cost(self):
        x=torch.randn(1,16,10,12);old=torch.randn(1,24,10,12)
        dense=self.cache.dense(x)
        for pattern in ('none','all','boundary','island'):
            mask=torch.zeros(1,1,10,12,dtype=torch.bool)
            if pattern=='all':mask.fill_(True)
            if pattern=='boundary':mask[:,:,:2,:2]=True;mask[:,:,-1,-1]=True
            if pattern=='island':mask[:,:,4:6,5:7]=True
            output,macs,support=self.cache.selected(x,old,mask)
            self.assertTrue(torch.allclose(output.masked_select(mask),dense.masked_select(mask),atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.equal(output.masked_select(~mask),old.masked_select(~mask)))
            expected=support*(self.cache.first.weight.numel()+self.cache.project.weight.numel())+int(mask.sum())*self.cache.second.weight.numel()
            self.assertEqual(macs,expected)
            if pattern=='none':self.assertEqual(macs,0)

    @torch.no_grad()
    def test_causal_cache_reuse_refresh_and_uncovered(self):
        x=torch.randn(1,16,10,12)
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        output,state,info=self.cache.step(x,None,meta)
        self.assertTrue(info['readout_refresh'])
        state['confidence']=torch.zeros(1,1,10,12)
        for stamp in (150000,200000):
            output,new,info=self.cache.step(x,state,dict(meta,timestamp_us=stamp))
            self.assertFalse(info['readout_refresh'])
            self.assertEqual(info['readout_extra_linear_macs'],0)
            self.assertTrue(torch.equal(output,state['value']))
            new['confidence']=state['confidence'];state=new
        _,new,info=self.cache.step(x,state,dict(meta,timestamp_us=250000))
        self.assertEqual(info['readout_refresh_reason'],'period')
        with self.assertRaises(ValueError):self.cache.step(x,state,dict(meta,timestamp_us=200000))
        _,_,info=self.cache.step(x,state,dict(meta,segment='b',timestamp_us=250000))
        self.assertEqual(info['readout_refresh_reason'],'reset')
        _,_,info=self.cache.step(x,state,dict(meta,timestamp_us=400000))
        self.assertEqual(info['readout_refresh_reason'],'age')
        _,initial,_=self.cache.step(x,None,meta);initial['confidence']=torch.zeros(1,1,10,12)
        moved=np.eye(4);moved[0,3]=1.
        _,_,info=self.cache.step(x,initial,dict(meta,timestamp_us=150000,pose=moved))
        self.assertGreater(info['readout_uncovered_fraction'],0)
        self.assertGreater(info['readout_write_fraction'],0)

    @torch.no_grad()
    def test_period1_exact_and_own_confidence_guard(self):
        x=torch.randn(1,16,10,12);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        cache=LocalReadoutCache(self.model,self.bounds,.25,period=1)
        output,state,_=cache.step(x,None,meta)
        state['confidence']=torch.zeros(1,1,10,12)
        y=x+1
        output,_,info=cache.step(y,state,dict(meta,timestamp_us=150000))
        self.assertTrue(torch.equal(output,cache.dense(y)))
        self.assertTrue(info['readout_refresh'])
        state['confidence'][:,:,5,6]=.5
        _,_,info=self.cache.step(x,state,dict(meta,timestamp_us=150000))
        self.assertGreater(info['readout_write_fraction'],0)
        self.assertTrue(self.cache.last_mask[0,0,5,6])

    @torch.no_grad()
    def test_bounded_period5_changes_only_readout_refresh(self):
        cache=LocalReadoutCache(self.model,self.bounds,.5,period=5)
        x=torch.randn(1,16,10,12);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,initial,_=cache.step(x,None,meta)
        initial['confidence']=torch.zeros(1,1,10,12);state=initial
        for index in range(1,6):
            _,new,info=cache.step(x,state,dict(meta,timestamp_us=100000+index*50000))
            self.assertEqual(info['readout_refresh'],index==5)
            self.assertLess(info['readout_full_age_s'],.45)
            new['confidence']=state['confidence'];state=new
        _,_,info=cache.step(x,initial,dict(meta,timestamp_us=550000))
        self.assertEqual(info['readout_refresh_reason'],'age')
        self.assertEqual(self.cache.max_full_age_us,250000)
        self.assertEqual(cache.max_full_age_us,450000)

    @torch.no_grad()
    def test_same_input_masked_dense_and_retained_reference(self):
        masked=LocalReadoutCache(self.model,self.bounds,.25,execution='masked_dense')
        x=torch.randn(1,16,10,12);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=self.cache.step(x,None,meta);state['confidence']=torch.zeros(1,1,10,12)
        y=x.clone();y[:,:,1,1]+=10
        meta=dict(meta,timestamp_us=150000)
        a,sa,ia=self.cache.step(y,state,meta)
        b,sb,ib=masked.step(y,state,meta)
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(self.cache.last_mask,masked.last_mask))
        self.assertTrue(torch.equal(sa['reference'],sb['reference']))
        self.assertTrue(torch.equal(sa['reference'].masked_select(~self.cache.last_mask),state['reference'].masked_select(~self.cache.last_mask)))
        self.assertGreater(ia['readout_extra_linear_macs'],0)
        self.assertEqual(ib['readout_extra_linear_macs'],0)


if __name__=='__main__':unittest.main()

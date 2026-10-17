import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory, warp_bev
from opencood.models.sub_modules.int_early_cache import EarlyCacheDetector, refresh_reason


class Stage(nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self,x):
        self.calls+=1
        return F.avg_pool2d(x,2)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.resnet=nn.Module();self.calls=0
        for i in range(3):setattr(self.resnet,'layer%d'%i,Stage())
    def decode_multiscale_feature(self,stages):
        self.calls+=1
        return sum(F.interpolate(s,size=stages[0].shape[-2:],mode='nearest') for s in stages)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_memory=INTFeatureMemory('gru',64,[-8,-8,-1,8,8,1])
        self.scatter=nn.Identity();self.scatter.model_cfg={'grid_size':[16,16,1]}
        self.backbone=Backbone();self.cls_head=nn.Conv2d(64,1,1);self.reg_head=nn.Conv2d(64,7,1)
        self.shrink_flag=False;self.use_dir=False
    def pillar_vfe(self,batch):
        return {'spatial_features':batch['voxel_features']}


class EarlyCacheTests(unittest.TestCase):
    def test_gru_observer_consumes_existing_warp_and_preserves_predictions(self):
        torch.manual_seed(3)
        fm=INTFeatureMemory('gru',16,[-8,-8,-1,8,8,1]).eval()
        x=torch.randn(1,16,16,16)
        meta=dict(segment='s',timestamp_us=0,pose=np.eye(4))
        with torch.no_grad():
            _,state,_=fm.step(x,None,meta)
            moved=np.eye(4);moved[0,3]=1
            meta=dict(meta,pose=moved,timestamp_us=100000)
            expected,next_state,_=fm.step(x,state,meta)
            observed=[]
            with patch('opencood.models.sub_modules.int_feature_memory.warp_bev',wraps=warp_bev) as warp:
                actual,current,_=fm.step(x,state,meta,state_observer=lambda *v:observed.append(v))
                self.assertEqual(warp.call_count,1)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(current['value'],next_state['value']))
            aligned,valid=warp_bev(state['value'],state['pose'],meta['pose'],fm.lidar_range)
            self.assertTrue(torch.equal(observed[0][1],aligned[:,:16]))
            self.assertTrue(torch.equal(observed[0][0],current['value'][:,:16]))
            self.assertTrue(torch.equal(observed[0][2],valid))

    def test_four_combinations_skip_actual_modules_and_force_refresh_after_one_reuse(self):
        for gate in ['gru','stage1']:
            for kind in ['stages','decoded']:
                m=Tiny().eval();cfg={'model':{'args':{'lidar_range':[-8,-8,-1,8,8,1]}}}
                wrapper=EarlyCacheDetector(m,cfg,gate,kind,threshold=100.)
                data={'processed_lidar':{'voxel_features':torch.randn(1,64,16,16),
                       'voxel_coords':torch.zeros(1,4),'voxel_num_points':torch.ones(1)}}
                meta=dict(segment='s',timestamp_us=0,pose=np.eye(4))
                with torch.no_grad():
                    _,state,_=wrapper.step(data,None,meta)
                    anchor=state['cache']['values']
                    before=[getattr(m.backbone.resnet,'layer%d'%i).calls for i in range(3)]
                    dec=m.backbone.calls
                    _,state,info=wrapper.step(data,state,dict(meta,timestamp_us=100000))
                    self.assertIsNone(info['early_refresh_reason'])
                    actual=[getattr(m.backbone.resnet,'layer%d'%i).calls-before[i] for i in range(3)]
                    self.assertEqual(actual,[int(gate=='stage1'),0,0])
                    self.assertEqual(m.backbone.calls-dec,int(kind=='stages'))
                    self.assertIs(state['cache']['values'],anchor)
                    _,state,info=wrapper.step(data,state,dict(meta,timestamp_us=200000))
                    self.assertEqual(info['early_refresh_reason'],'period')
                    _,_,info=wrapper.step(data,state,dict(meta,segment='new',timestamp_us=300000))
                    self.assertEqual(info['early_refresh_reason'],'initial')

    def test_guard_priority_and_invalid_policy(self):
        self.assertEqual(refresh_reason(False,1,.1,False,.8,.5),'change')
        self.assertEqual(refresh_reason(False,1,.1,False,.2,.5),'coverage')
        self.assertEqual(refresh_reason(False,2,.1,False,None,.5),'period')
        self.assertEqual(refresh_reason(False,1,.3,False,None,.5),'age')
        with self.assertRaises(ValueError):refresh_reason(False,1,0,True,0,.5)
        with self.assertRaises(ValueError):
            EarlyCacheDetector(Tiny().eval(),{'model':{'args':{'lidar_range':[-8,-8,-1,8,8,1]}}},period=3)


if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.codyntrust_downsample_conv import DownsampleConv
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector


def make_pair():
    teacher=SimpleNamespace(pillar_vfe=nn.Identity(),scatter=nn.Identity(),use_dir=True,shrink_flag=True,
        backbone=ResNetBEVBackbone(dict(layer_nums=[3,4,5],layer_strides=[2,2,2],
            num_filters=[128,256,512],upsample_strides=[1,2,4],num_upsample_filter=[128,128,128])),
        shrink_conv=DownsampleConv(dict(input_dim=384,kernal_size=[3],dim=[256],stride=[1],padding=[1])),
        cls_head=nn.Conv2d(256,2,1),reg_head=nn.Conv2d(256,14,1),dir_head=nn.Conv2d(256,4,1))
    for m in (teacher.backbone,teacher.shrink_conv):m.eval()
    model=PyramidStateDetector(teacher,{'model':{'args':{'lidar_range':[-4,-4,-1,4,4,1]}}})
    return teacher,model


class PyramidTests(unittest.TestCase):
    def test_complete_spatial_path_is_preserved_at_initialization(self):
        torch.manual_seed(303)
        teacher,model=make_pair()
        self.assertGreater(sum(p.numel() for p in model.parameters()),20_000_000)
        self.assertEqual([len(model.core.observation),len(model.core.coarse_encoder),len(model.core.deep_encoder)],[3,4,5])
        for target,source in model.migrated_source_pairs(teacher):
            self.assertTrue(all(torch.equal(v,source.state_dict()[k]) for k,v in target.state_dict().items()))
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        with torch.no_grad():
            features=teacher.shrink_conv(teacher.backbone({'spatial_features':x})['spatial_features_2d'])
            expected=dict(psm=teacher.cls_head(features),rm=teacher.reg_head(features),dm=teacher.dir_head(features))
            actual,state,info=model.step_spatial(x,None,meta)
        self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
        self.assertEqual([tuple(state[k].shape) for k in model.state_keys],
                         [(1,128,12,16),(1,256,6,8),(1,512,3,4)])
        self.assertEqual(info['state_bytes'],sum(state[k].numel()*4 for k in model.state_keys))
        self.assertTrue(model.updates_are_initial_identity())

    def test_all_three_histories_receive_gradients_and_detach_at_window_boundary(self):
        from opencood.tools.run_int_two_scale import detach_stream_state
        torch.manual_seed(303)
        _,model=make_pair();model.train()
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
        for name in ('fine_update','coarse_update','deep_update'):
            nn.init.normal_(getattr(model.core,name).history.weight,std=.001)
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta,False)
        before={key:state[key].detach().clone() for key in model.state_keys}
        for key in model.state_keys:state[key].retain_grad()
        pred,later,_=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
        pred['psm'].square().mean().backward()
        for key in model.state_keys:
            self.assertGreater(float(state[key].grad.abs().sum()),0)
            self.assertTrue(torch.equal(state[key].detach(),before[key]))
        detached=detach_stream_state(later)
        self.assertTrue(all(detached[k].grad_fn is None for k in model.state_keys))
        self.assertEqual(detached['timestamp_us'],200000)
        self.assertIs(detached['pose'],meta['pose'])

    def test_reset_scene_gap_and_optimizer_include_deep_path(self):
        from opencood.tools.run_int_two_scale import make_optimizer
        _,model=make_pair()
        optimizer=make_optimizer(model,'transfer');groups={g['name']:g for g in optimizer.param_groups}
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.core.parameters()})
        self.assertTrue({id(p) for p in model.core.deep_encoder.parameters()} <= {id(p) for p in groups['coarse_encoder']['params']})
        self.assertTrue({id(p) for p in model.core.deblocks.parameters()} <= {id(p) for p in groups['decoder']['params']})
        self.assertEqual(groups['decoder']['lr'],1e-5)
        for name in ('fine_update','coarse_update','deep_update'):
            nn.init.constant_(getattr(model.core,name).history.weight,.001)
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta)
            for key in model.state_keys:state[key]=torch.ones_like(state[key])
            next_meta=dict(meta,timestamp_us=200000)
            expected,_,_=model.step_spatial(x,None,next_meta)
            recurrent,_,_=model.step_spatial(x,state,next_meta)
            self.assertTrue(any(not torch.equal(recurrent[k],expected[k]) for k in expected))
            for changed,reason in [(dict(next_meta,reset_before=True),'explicit'),
                                   (dict(next_meta,segment='b'),'scene'),
                                   (dict(next_meta,timestamp_us=500000),'gap')]:
                actual,_,info=model.step_spatial(x,state,changed)
                self.assertEqual(info['reset'],reason)
                self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
            model.memory_policy='reset'
            actual,_,_=model.step_spatial(x,state,next_meta)
            self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
            model.memory_policy='recurrent'
            with self.assertRaises(ValueError):model.step_spatial(x,state,meta)

    def test_reset_probe_has_zero_history_gradients_and_learns_current_path(self):
        torch.manual_seed(303)
        _,model=make_pair();model.memory_policy='reset'
        for name in ('fine_update','coarse_update','deep_update'):
            nn.init.normal_(getattr(model.core,name).history.weight,std=.001)
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta,False)
        prediction,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
        self.assertEqual(info['reset'],'explicit')
        prediction['psm'].square().mean().backward()
        for name in ('fine_update','coarse_update','deep_update'):
            gradient=getattr(model.core,name).history.weight.grad
            self.assertIsNotNone(gradient)
            self.assertEqual(float(gradient.abs().sum()),0.)
        self.assertGreater(float(model.core.cls_head.weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()

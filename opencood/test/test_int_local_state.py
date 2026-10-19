import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_local_state import LocalStateDetector, feature_distillation


def source_teacher():
    def block(inputs, outputs, stride):
        down = nn.Sequential(nn.Conv2d(inputs,outputs,1,stride=stride,bias=False),nn.BatchNorm2d(outputs))
        return BasicBlock(inputs,outputs,stride,down)
    teacher=nn.Module()
    teacher.pillar_vfe=nn.Linear(4,64)
    teacher.scatter=nn.Identity()
    teacher.backbone=nn.Module(); teacher.backbone.use_dropout=False
    teacher.backbone.resnet=nn.Module()
    teacher.backbone.resnet.layer0=nn.Sequential(block(64,128,2))
    teacher.backbone.resnet.layer1=nn.Sequential(block(128,256,2),BasicBlock(256,256))
    teacher.cls_head=nn.Conv2d(256,2,1); teacher.reg_head=nn.Conv2d(256,14,1)
    teacher.dir_head=nn.Conv2d(256,4,1); teacher.use_dir=True
    return teacher.eval()


class LocalStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(303)
        self.teacher=source_teacher()
        self.cfg=dict(model=dict(args=dict(lidar_range=[-12,-8,-1,12,8,1])))

    def meta(self,t=0,scene='s'):
        return dict(timestamp_us=t,segment=scene,pose=torch.eye(4).numpy())

    def test_migration_identity_projection_and_no_shared_parameters(self):
        model=LocalStateDetector(self.teacher,self.cfg)
        x=torch.randn(1,385,9,13)
        self.assertTrue(torch.equal(model.update.input_projection(x),x[:,:128]))
        self.assertEqual(self.teacher.backbone.resnet.layer1[0].conv1.stride,(2,2))
        self.assertEqual(model.update.blocks[0].conv1.stride,(1,1))
        self.assertEqual(model.update.blocks[0].downsample[0].stride,(1,1))
        for i in range(2):
            original=self.teacher.backbone.resnet.layer1[i].state_dict()
            self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.update.blocks[i].state_dict().items()))
        source_ids={id(p) for p in self.teacher.parameters()}
        self.assertFalse(any(id(p) in source_ids for p in model.parameters()))
        self.assertFalse(hasattr(model,'feature_memory')); self.assertFalse(hasattr(model,'backbone'))

    def test_stage_a_ignores_old_state_and_shapes(self):
        model=LocalStateDetector(self.teacher,self.cfg,'reset')
        with torch.no_grad():
            x=torch.randn(1,64,24,32)
            pred,state,_=model.step_spatial(x,None,self.meta())
            modified=dict(state,value=state['value']+100)
            actual,next_state,info=model.step_spatial(x,modified,self.meta(100000))
            self.assertTrue(all(torch.equal(actual[k],pred[k]) for k in pred))
            self.assertEqual(tuple(state['value'].shape),(1,256,12,16))
            self.assertEqual(info['reset'],'single_frame_stage_a')

    def test_fixed_bn_and_real_gradients(self):
        for p in self.teacher.parameters():p.requires_grad_(False)
        model=LocalStateDetector(self.teacher,self.cfg).train()
        before={k:v.clone() for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
        prediction,state,_=model.step_spatial(torch.randn(1,64,16,24),None,self.meta(),detach_state=False)
        loss=sum(v.square().mean() for v in prediction.values())
        loss.backward()
        self.assertGreater(float(model.update.input_projection.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.observation[0].conv1.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.pillar_vfe.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_recurrent_gradient_detach_and_time_guards(self):
        model=LocalStateDetector(self.teacher,self.cfg,'recurrent').train()
        with torch.no_grad():model.update.input_projection.weight[:,128:384].fill_(.0001)
        x=torch.randn(1,64,16,24)
        _,first,_=model.step_spatial(x,None,self.meta(),detach_state=False)
        first['value'].retain_grad()
        saved=first['value'].detach().clone()
        _,second,_=model.step_spatial(x,first,self.meta(100000),detach_state=False)
        second['value'].mean().backward()
        self.assertGreater(float(first['value'].grad.abs().sum()),0)
        self.assertTrue(torch.equal(first['value'].detach(),saved))
        _,detached,_=model.step_spatial(x,None,self.meta())
        self.assertFalse(detached['value'].requires_grad)
        with self.assertRaises(ValueError):model.step_spatial(x,detached,self.meta())
        _,_,info=model.step_spatial(x,detached,self.meta(300000))
        self.assertEqual(info['reset'],'gap')
        _,_,info=model.step_spatial(x,detached,self.meta(0,'new'))
        self.assertEqual(info['reset'],'scene')

    def test_radius_four_crop_matches_dense_at_boundaries(self):
        model=LocalStateDetector(self.teacher,self.cfg)
        # Nonzero fixed BN offsets catch invalid external zero padding.
        for m in model.update.modules():
            if isinstance(m,nn.BatchNorm2d):m.bias.data.uniform_(.1,.2)
        with torch.no_grad():
            x=torch.randn(1,385,33,43)
            ref=model.update(x); old=torch.randn_like(ref)
            mask=torch.zeros_like(ref[:,:1],dtype=torch.bool)
            mask[...,0,0]=True; mask[...,-1,-1]=True; mask[...,16,21]=True
            actual,written,_=refresh_tiles(model.update,x,old,mask,1,4,tile=2,allow_fallback=False)
            selected=written.expand_as(actual)
            self.assertTrue(torch.allclose(actual[selected],ref[selected],atol=1e-5,rtol=1e-4))
            self.assertTrue(torch.equal(actual[~selected],old[~selected]))

    def test_feature_distillation_is_finite_and_teacher_detached(self):
        student=torch.randn(1,256,5,7,requires_grad=True)
        teacher=torch.zeros_like(student,requires_grad=True)
        logits=torch.randn(1,2,5,7,requires_grad=True)
        loss=feature_distillation(student,teacher,dict(psm=logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(float(student.grad.abs().sum()),0)
        self.assertIsNone(teacher.grad); self.assertIsNone(logits.grad)


if __name__=='__main__':unittest.main()

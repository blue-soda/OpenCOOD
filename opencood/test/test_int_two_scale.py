import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_two_scale import TwoScaleDetector


def block(ci, co, stride):
    return BasicBlock(ci,co,stride,nn.Sequential(nn.Conv2d(ci,co,1,stride,bias=False),nn.BatchNorm2d(co)))


class TwoScaleTests(unittest.TestCase):
    def model(self, policy='recurrent', architecture='two_scale_v1'):
        teacher=SimpleNamespace(pillar_vfe=nn.Identity(),scatter=nn.Identity(),use_dir=False,
            cls_head=nn.Conv2d(256,2,1),reg_head=nn.Conv2d(256,14,1),
            backbone=SimpleNamespace(resnet=SimpleNamespace(layer0=nn.Sequential(block(64,128,2),BasicBlock(128,128),BasicBlock(128,128)),
                layer1=nn.Sequential(block(128,256,2),BasicBlock(256,256)))))
        return TwoScaleDetector(teacher,{'model':{'args':{'lidar_range':[-4,-4,-1,4,4,1]}}},policy,architecture)

    def test_encoder_depth_pair_preserves_shared_initialization_and_state_contract(self):
        torch.manual_seed(303)
        shallow=self.model(architecture='two_scale_narrow')
        torch.manual_seed(303)
        full=self.model(architecture='two_scale_full')
        self.assertEqual(len(full.core.observation),3)
        self.assertIsInstance(shallow.core.observation,BasicBlock)
        self.assertNotEqual(shallow.version,full.version)
        common={k:v for k,v in shallow.state_dict().items() if not k.startswith('core.observation.')}
        self.assertTrue(all(torch.equal(v,full.state_dict()[k]) for k,v in common.items()))
        self.assertTrue(all(torch.equal(v,full.core.observation[0].state_dict()[k])
                            for k,v in shallow.core.observation.state_dict().items()))
        x=torch.randn(1,64,16,20);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        for model in (shallow,full):
            model.train()
            self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
            pred,state,_=model.step_spatial(x,None,meta,False)
            self.assertEqual(pred['psm'].shape,(1,2,8,10))
            self.assertEqual(state['coarse'].shape,(1,256,4,5))
            _,later,_=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
            reset=copy.deepcopy(model);reset.memory_policy='reset'
            expected,_,_=reset.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
            actual,_,_=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
            self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
            actual['psm'].square().mean().backward()
            self.assertGreater(float(model.core.readout.projection.weight.grad.abs().sum()),0)

    def test_transfer_optimizer_covers_each_parameter_once_and_preserves_lr_ratios(self):
        from opencood.tools.run_int_two_scale import make_optimizer
        model=self.model(architecture='two_scale_full')
        optimizer=make_optimizer(model,'transfer')
        groups={g['name']:g for g in optimizer.param_groups}
        self.assertEqual(set(groups),{'new','observation','coarse_encoder','heads'})
        for name in ('observation','coarse_encoder'):
            self.assertEqual({id(p) for p in groups[name]['params']},
                             {id(p) for p in getattr(model.core,name).parameters()})
        all_ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(all_ids),len(set(all_ids)))
        self.assertEqual(set(all_ids),{id(p) for p in model.core.parameters()})
        self.assertAlmostEqual(groups['new']['lr']/groups['observation']['lr'],30.)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=0,
            min_lr=[g['lr']/30 for g in optimizer.param_groups])
        scheduler.step(.5);scheduler.step(.4)
        self.assertAlmostEqual(groups['new']['lr'],.00015)
        self.assertAlmostEqual(groups['new']['lr']/groups['heads']['lr'],30.)
        self.assertEqual(len(make_optimizer(model).param_groups),1)

    def test_medium_context_preserves_initial_function_and_learns_through_new_branch(self):
        from opencood.tools.run_int_two_scale import make_optimizer
        torch.manual_seed(303)
        small=self.model(architecture='two_scale_full')
        torch.manual_seed(303)
        medium=self.model(architecture='two_scale_medium')
        self.assertTrue(all(torch.equal(v,medium.state_dict()[k]) for k,v in small.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in medium.parameters())-
                         sum(p.numel() for p in small.parameters()),3419584)
        optimizer=make_optimizer(medium,'transfer')
        new_ids={id(p) for g in optimizer.param_groups if g['name']=='new' for p in g['params']}
        self.assertTrue({id(p) for p in medium.core.coarse_context.parameters()} <= new_ids)
        # Odd spatial dimensions exercise resize back to the exact coarse lattice.
        x=torch.randn(1,64,18,22);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        a,sa,ia=small.step_spatial(x,None,meta,False)
        b,sb,ib=medium.step_spatial(x,None,meta,False)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        self.assertTrue(all(torch.equal(sa[k],sb[k]) for k in ('value','coarse')))
        self.assertEqual(ia['state_bytes'],ib['state_bytes'])
        context=medium.core.coarse_context
        b['psm'].square().mean().backward()
        self.assertGreater(float(context.projection.weight.grad.abs().sum()),0)
        optimizer.step();optimizer.zero_grad()
        b,_,_=medium.step_spatial(x,None,meta,False)
        b['psm'].square().mean().backward()
        self.assertGreater(float(context.down.weight.grad.abs().sum()),0)
        for block in context.blocks:
            self.assertGreater(float(block.residual[-1].weight.grad.abs().sum()),0)

    def test_initial_policy_equivalence_and_state_ownership(self):
        torch.manual_seed(303)
        model=self.model();reset=copy.deepcopy(model);reset.memory_policy='reset'
        x=torch.randn(1,64,16,20);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta)
        before={k:v.clone() for k,v in state.items() if torch.is_tensor(v)}
        meta=dict(meta,timestamp_us=200000)
        a,_,_=model.step_spatial(x,state,meta);b,_,_=reset.step_spatial(x,state,meta)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        self.assertTrue(all(torch.equal(state[k],v) for k,v in before.items()))
        self.assertEqual(state['value'].shape,(1,128,8,10))
        self.assertEqual(state['coarse'].shape,(1,256,4,5))
        with self.assertRaises(ValueError):model.step_spatial(x,state,dict(meta,timestamp_us=100000))
        _,_,info=model.step_spatial(x,state,dict(meta,timestamp_us=500000))
        self.assertEqual(info['reset'],'gap')

    def test_temporal_gradients_and_detach(self):
        model=self.model().train()
        nn.init.normal_(model.core.fine_update.history.weight,std=.01)
        nn.init.normal_(model.core.coarse_update.history.weight,std=.01)
        x=torch.randn(1,64,16,20);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta,False)
        for k in ('value','coarse'):state[k].retain_grad()
        pred,_,_=model.step_spatial(x,state,dict(meta,timestamp_us=200000),False)
        pred['psm'].square().mean().backward()
        for k in ('value','coarse'):self.assertGreater(float(state[k].grad.abs().sum()),0)
        _,detached,_=model.step_spatial(x,None,meta,True)
        self.assertTrue(all(detached[k].grad_fn is None for k in ('value','coarse')))

    def test_reset_intervention_ignores_both_nonzero_states_without_changing_weights(self):
        torch.manual_seed(303)
        model=self.model()
        nn.init.constant_(model.core.fine_update.history.weight,.01)
        nn.init.constant_(model.core.coarse_update.history.weight,.01)
        x=torch.randn(1,64,16,20)
        meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        _,state,_=model.step_spatial(x,None,meta)
        state=dict(state,value=torch.ones_like(state['value']),coarse=torch.ones_like(state['coarse']))
        meta=dict(meta,timestamp_us=200000)
        expected,_,_=model.step_spatial(x,None,meta)
        recurrent,_,_=model.step_spatial(x,state,meta)
        before={k:v.clone() for k,v in model.state_dict().items()}
        model.memory_policy='reset'
        actual,_,info=model.step_spatial(x,state,meta)
        self.assertEqual(info['reset'],'explicit')
        self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
        self.assertTrue(any(not torch.equal(recurrent[k],expected[k]) for k in recurrent))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))


if __name__=='__main__':unittest.main()

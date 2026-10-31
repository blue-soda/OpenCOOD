import unittest
import numpy as np
import torch
from torch import nn
from opencood.test.test_int_pyramid_state import make_pair
from opencood.models.sub_modules.int_observation_bank import ObservationBankDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev


class ObservationBankTests(unittest.TestCase):
    def test_initial_identity_matches_complete_source_and_parameter_budget(self):
        torch.manual_seed(303);teacher,model=make_pair(ObservationBankDetector)
        self.assertEqual(sum(p.numel() for p in model.core.evidence_fusion.parameters()),329408)
        self.assertFalse(hasattr(model.core,'fine_update'))
        for target,source in model.migrated_source_pairs(teacher):
            self.assertTrue(all(torch.equal(v,source.state_dict()[k]) for k,v in target.state_dict().items()))
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        with torch.no_grad():
            features=teacher.shrink_conv(teacher.backbone({'spatial_features':x})['spatial_features_2d'])
            expected=dict(psm=teacher.cls_head(features),rm=teacher.reg_head(features),dm=teacher.dir_head(features))
            state=None
            for timestamp,count in [(100000,0),(200000,1),(300000,2)]:
                actual,state,info=model.step_spatial(x,state,dict(meta,timestamp_us=timestamp))
                self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))
                self.assertEqual(info['history_observations_used'],count)
                self.assertEqual(info['state_bytes'],x.numel()*x.element_size()*2)

    def test_raw_bank_own_pose_age_reset_and_causality(self):
        _,model=make_pair(ObservationBankDetector)
        x=torch.randn(1,64,24,32,requires_grad=True)
        a=dict(segment='a',timestamp_us=100000,pose=np.eye(4));pose2=np.eye(4);pose2[0,3]=.5
        b=dict(a,timestamp_us=200000,pose=pose2)
        with torch.no_grad():
            _,first,_=model.step_spatial(x,None,a)
            _,second,_=model.step_spatial(2*x,first,b)
        self.assertTrue(torch.equal(second['value'],2*x))
        self.assertTrue(torch.equal(second['older'],x))
        self.assertFalse(second['value'].requires_grad)
        self.assertFalse(second['older'].requires_grad)
        current=dict(a,timestamp_us=300000)
        old,masks,ages,count=model.aligned_history(x,second,current,None)
        self.assertEqual(count,2)
        for index,key,pose,age in [(0,'value',pose2,.1),(1,'older',a['pose'],.2)]:
            expected,valid=warp_bev(second[key],pose,current['pose'],model.bounds)
            self.assertTrue(torch.equal(old[index],expected))
            self.assertTrue(torch.allclose(ages[index],valid.float()*(age/.25)))
        late=dict(current,timestamp_us=450000)
        self.assertEqual(model.aligned_history(x,second,late,None)[-1],1)
        with torch.no_grad():
            for changed,reason in [(dict(current,segment='b'),'scene'),
                                   (dict(current,reset_before=True),'explicit'),
                                   (dict(current,timestamp_us=500001),'gap')]:
                _,state,info=model.step_spatial(x,second,changed)
                self.assertEqual(info['reset'],reason)
                self.assertEqual(info['history_observations_used'],0)
                self.assertIsNone(state['older_timestamp_us'])
            with self.assertRaises(ValueError):model.step_spatial(x,second,b)
            with self.assertRaises(ValueError):model.aligned_history(x,dict(second,older_timestamp_us=400000),current,None)

    def test_fusion_learns_history_and_reset_removes_it(self):
        from opencood.tools.run_int_two_scale import make_optimizer
        torch.manual_seed(303);_,model=make_pair(ObservationBankDetector)
        groups=make_optimizer(model,'transfer').param_groups
        new_ids={id(p) for g in groups if g['name']=='new' for p in g['params']}
        self.assertEqual(new_ids,{id(p) for p in model.core.evidence_fusion.parameters()})
        fusion=model.core.evidence_fusion;optimizer=torch.optim.AdamW(fusion.parameters(),lr=.001)
        evidence=torch.randn(1,196,24,32)
        (fusion(evidence)-1).square().mean().backward()
        self.assertGreater(float(fusion[-1].weight.grad.abs().sum()),0.)
        optimizer.step();optimizer.zero_grad()
        fusion(evidence).square().mean().backward()
        self.assertGreater(float(fusion[0].weight.grad.abs().sum()),0.)
        x=torch.randn(1,64,24,32);meta=dict(segment='a',timestamp_us=100000,pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta)
            current=dict(meta,timestamp_us=200000)
            recurrent,_,_=model.step_spatial(x,state,current)
            empty,_,_=model.step_spatial(x,None,current)
            self.assertTrue(any(not torch.equal(recurrent[k],empty[k]) for k in empty))
            model.memory_policy='reset'
            reset,_,info=model.step_spatial(x,state,current)
            self.assertTrue(all(torch.equal(reset[k],empty[k]) for k in empty))
            self.assertEqual(info['history_observations_used'],0)


if __name__=='__main__':unittest.main()

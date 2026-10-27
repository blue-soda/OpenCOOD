import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_y_only import YOnlyUpdater,YOnlyState


class AddY(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,evidence,old,valid):return old+self.scale*evidence


class TestYOnly(unittest.TestCase):
    def test_paired_evaluator_accepts_whole_path_cost_without_readout_ratio(self):
        from opencood.tools.train_int_recurrent_deep import evaluate
        import opencood.tools.train_int_recurrent_deep as runner
        class Mode:
            def step(self,data,state,meta):
                return {'psm':torch.ones(1)},None,dict(whole_model_macs=123,
                    deep_partial=True,local_partial=True,spatial_refresh=False,
                    state_bytes=4,spatial_state_bytes=0,anchor_bytes=0,readout_state_bytes=16,
                    readout_refresh=False,readout_write_fraction=1.)
        rows=[dict(supervised=False)]*12
        stream=[dict(ego={},meta=dict(supervised=False)) for _ in rows]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner,'loader',return_value=(None,stream)), \
                patch.object(runner,'to_device',side_effect=lambda x,d:x), \
                patch.object(runner,'sample_gpu_usage',return_value={'timing_valid':True}), \
                patch.object(torch.cuda,'synchronize'), \
                patch.object(runner.eval_utils,'calculate_ap',return_value=(.7553159775482435,)):
            report=evaluate(SimpleNamespace(gpu=0),Mode(),None,{},rows,Path(directory),{},0,0,
                            variants={'y_only':Mode()},share_gpu=True)
        result=report['results']['y_only']
        self.assertIsNone(result['mean_readout_executed_fraction'])
        self.assertEqual(result['mean_readout_write_fraction'],1.)
        self.assertEqual(result['conv_linear_gflops'],246/1e9)
        self.assertEqual(result['mean_state_bytes'],20.)

    def test_own_recurrence_no_ordinary_full_call_and_teacher_isolation(self):
        cache=YOnlyState(AddY(),[-2,-2,-1,2,2,1],3)
        evidence=torch.ones(1,1,4,4);calls=[]
        def full():calls.append(1);return torch.full_like(evidence,10.)
        meta=dict(segment='a',timestamp_us=1,pose=np.eye(4))
        _,state,_=cache.step(evidence,None,meta,full)
        saved=state['value'].clone();next_meta=dict(meta,timestamp_us=50001)
        first,new,info=cache.step(evidence,state,next_meta,full,supervise=True)
        self.assertEqual(len(calls),2);self.assertEqual(float(first.mean()),11.)
        self.assertTrue(torch.equal(saved,state['value']))
        no_teacher,_,_=cache.step(evidence,state,next_meta,lambda: self.fail('Full path called'))
        self.assertTrue(torch.equal(first,no_teacher))
        second,new,_=cache.step(evidence,new,dict(meta,timestamp_us=100001),full,supervise=True)
        second.sum().backward();self.assertEqual(float(cache.updater.scale.grad),32.)
        self.assertEqual(float(second.mean()),12.)
        value,_,info=cache.step(evidence,new,dict(meta,timestamp_us=150001),full)
        self.assertTrue(info['y_refresh']);self.assertEqual(float(value.mean()),10.)
        with self.assertRaises(ValueError):cache.step(evidence,new,dict(meta,timestamp_us=100001),full)
        _,_,info=cache.step(evidence,new,dict(meta,timestamp_us=300001),full)
        self.assertEqual(info['y_refresh_reason'],'age')
        _,_,info=cache.step(evidence,new,dict(meta,segment='b'),full)
        self.assertEqual(info['y_refresh_reason'],'reset')

    def test_zero_coverage_never_directly_copies_old_and_local_dependency(self):
        torch.manual_seed(303);net=YOnlyUpdater(3,4,8)
        evidence=torch.randn(1,3,11,13);old=torch.randn(1,4,11,13);valid=torch.zeros(1,1,11,13,dtype=torch.bool)
        hidden=net.features(torch.cat((evidence,old,valid.float()),1))
        self.assertTrue(torch.equal(net(evidence,old,valid),torch.relu(net.candidate(hidden))))
        # Two3x3 layers: changing input outside radius2 cannot affect the center.
        valid[:]=True;base=net(evidence,old,valid)
        changed=evidence.clone();changed[:,:,0,:]+=100
        other=net(changed,old,valid)
        self.assertTrue(torch.equal(base[:,:,5,6],other[:,:,5,6]))
        base.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_period_five_and_warp_boundary(self):
        cache=YOnlyState(AddY(),[-2,-2,-1,2,2,1],5)
        evidence=torch.ones(1,1,4,4);meta=dict(segment='a',timestamp_us=1,pose=np.eye(4));state=None
        flags=[]
        for i in range(6):
            _,state,info=cache.step(evidence,state,dict(meta,timestamp_us=1+i*50000),lambda:torch.zeros_like(evidence))
            flags.append(info['y_refresh'])
        self.assertEqual(flags,[True,False,False,False,False,True])
        moved=np.eye(4);moved[0,3]=20
        _,_,info=cache.step(evidence,state,dict(meta,timestamp_us=300001,pose=moved),lambda:None)
        self.assertEqual(info['y_uncovered_fraction'],1.)

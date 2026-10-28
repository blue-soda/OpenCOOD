import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from opencood.models.sub_modules.int_z_only import ZOnlyUpdater,ZOnlyDetector,make_reader
from opencood.models.sub_modules.int_y_only import YOnlyState


class TinyDetector(ZOnlyDetector):
    def __init__(self,joint):
        self.model=nn.Module();self.model.cls_head=nn.Conv2d(4,2,1)
        self.model.reg_head=nn.Conv2d(4,7,1);self.model.use_dir=False
        self.readout=nn.Module();self.readout.spatial=nn.Conv2d(96,4,3,padding=1)
        self.readout.tail=nn.Sequential(nn.BatchNorm2d(4),nn.ReLU())
        self.readout.eval()
        for m in (self.model,self.readout):
            for p in m.parameters():p.requires_grad_(False)
        self.updater=ZOnlyUpdater(4);self.reader=make_reader(self.readout,joint)
        self.cache=YOnlyState(self.updater,[-2,-2,-1,2,2,1],3)
        self.execution_costs={};self.calls=0

    def front(self,data,state,meta):
        return data.detach(),0,dict(reset=state is None,state_bytes=0)

    @torch.no_grad()
    def full_from_evidence(self,evidence):
        self.calls+=1
        return evidence[:,:96].detach()


class TestZOnly(unittest.TestCase):
    def test_signed_candidate_and_local_dependencies(self):
        net=ZOnlyUpdater(4)
        with torch.no_grad():net.candidate.weight.zero_();net.candidate.bias.fill_(-2.)
        e=torch.ones(1,128,7,9);old=torch.zeros(1,96,7,9);valid=torch.zeros(1,1,7,9,dtype=torch.bool)
        self.assertTrue(torch.equal(net(e,old,valid),torch.full_like(old,-2.)))
        torch.manual_seed(3);net=ZOnlyUpdater(4);valid[:]=True
        base=net(e,old,valid);changed=e.clone();changed[:,:,0,:]+=50
        self.assertTrue(torch.equal(base[:,:,4,4],net(changed,old,valid)[:,:,4,4]))

    def test_reader_can_adapt_without_changing_teacher_or_bn_statistics(self):
        torch.manual_seed(3);mode=TinyDetector(True)
        target=torch.randn(1,96,7,9);frozen=mode.teacher_y(target).clone()
        self.assertTrue(torch.equal(mode.reader(target),frozen))
        for a,b in zip(mode.reader.parameters(),list(mode.readout.spatial.parameters())+list(mode.readout.tail.parameters())):
            self.assertNotEqual(a.data_ptr(),b.data_ptr())
        buffers={k:v.clone() for k,v in mode.reader.named_buffers()}
        opt=torch.optim.SGD(mode.reader.parameters(),lr=.01)
        mode.reader(target).square().mean().backward();opt.step()
        self.assertTrue(torch.equal(mode.teacher_y(target),frozen))
        self.assertFalse(torch.equal(mode.reader(target),frozen))
        self.assertTrue(all(torch.equal(buffers[k],v) for k,v in mode.reader.named_buffers()))
        self.assertTrue(all(p.grad is None for p in mode.readout.parameters()))

    def test_own_z_only_state_and_teacher_isolation(self):
        torch.manual_seed(4);mode=TinyDetector(True)
        e=torch.randn(1,128,7,9);meta=dict(segment='s',timestamp_us=1,pose=np.eye(4))
        _,first,_=mode.step(e,None,meta,supervise=True)
        previous=first['z']['value'].clone();meta=dict(meta,timestamp_us=50001)
        mode.calls=0
        with torch.no_grad():pred,state,info=mode.step(e,first,meta)
        self.assertEqual(mode.calls,0)
        other,supervised,details=mode.step(e,first,meta,supervise=True)
        self.assertEqual(mode.calls,1)
        self.assertTrue(all(torch.equal(pred[k],other[k]) for k in pred))
        self.assertEqual(set(state),{'fm','z'})
        self.assertTrue(torch.equal(state['z']['value'],supervised['z']['value']))
        self.assertTrue(torch.equal(previous,first['z']['value']))
        self.assertEqual(info['readout_state_bytes'],state['z']['value'].numel()*4)
        loss=other['rm'].square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in mode.updater.parameters()))
        self.assertTrue(all(p.grad is not None for p in mode.reader.parameters()))
        self.assertTrue(all(p.grad is None for p in mode.model.parameters()))
        self.assertTrue(all(p.grad is None for p in mode.readout.parameters()))

    def test_frozen_reader_still_backpropagates_into_updater(self):
        torch.manual_seed(4);mode=TinyDetector(False)
        e=torch.randn(1,128,7,9);meta=dict(segment='s',timestamp_us=1,pose=np.eye(4))
        _,state,_=mode.step(e,None,meta,supervise=True)
        pred,_,_=mode.step(e,state,dict(meta,timestamp_us=50001),supervise=True)
        pred['rm'].square().mean().backward()
        self.assertTrue(any(float(p.grad.abs().sum())>0 for p in mode.updater.parameters()))
        self.assertTrue(all(p.grad is None for p in mode.reader.parameters()))


class TestZDiagnosis(unittest.TestCase):
    def test_signed_counterfactuals_do_not_change_history(self):
        from opencood.tools.diagnose_int_z_only import observe_update
        torch.manual_seed(6);net=ZOnlyUpdater(4)
        with torch.no_grad():net.candidate.weight.zero_();net.candidate.bias.fill_(-2.)
        e=torch.randn(1,128,7,9);old=torch.randn(1,96,7,9);teacher=old+1
        valid=torch.ones(1,1,7,9,dtype=torch.bool);valid[:,:,:,0]=False
        snapshot=old.clone();before={k:v.clone() for k,v in net.state_dict().items()}
        values,gate=observe_update(net,e,old,teacher,valid)
        self.assertTrue(torch.equal(values['recursive'],net(e,old,valid)))
        self.assertTrue(torch.equal(values['candidate_only'],torch.full_like(old,-2.)))
        self.assertTrue(torch.equal(values['recursive'][:,:,:,0],torch.full_like(old[:,:,:,0],-2.)))
        self.assertTrue(torch.equal(old,snapshot))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in net.state_dict().items()))
        self.assertFalse(torch.equal(values['recursive'],values['teacher_history']))

    def test_crossed_reader_diagnoses_drift_without_changing_z(self):
        from opencood.tools.diagnose_int_z_only import observe_readers
        torch.manual_seed(7);mode=TinyDetector(False)
        z=torch.randn(1,96,7,9);target=z+1;snapshot=z.clone()
        original=observe_readers(mode,z,target)
        self.assertTrue(torch.equal(original['fixed_on_own'],original['adapted_on_own']))
        with torch.no_grad():mode.reader[0].bias.add_(2.)
        changed=observe_readers(mode,z,target)
        self.assertTrue(torch.equal(changed['fixed_on_own'],original['fixed_on_own']))
        self.assertFalse(torch.equal(changed['adapted_on_own'],original['adapted_on_own']))
        self.assertFalse(torch.equal(changed['adapted_on_full'],original['adapted_on_full']))
        self.assertTrue(torch.equal(z,snapshot))

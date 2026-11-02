import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_readout_refresh import ReadoutRefreshDetector
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_block_cache import block_lidar_range


class ReadoutRefreshTests(unittest.TestCase):
    def test_dense_and_period_one_exact_parent(self):
        base=fixture();models=[ReadoutRefreshDetector(base,'dense'),ReadoutRefreshDetector(base,period=1)]
        states=[None]*3;calls=[]
        handles=[m.core.fast_readout.register_forward_hook(lambda *_:calls.append(1)) for m in models]
        with torch.no_grad():
            for i in range(4):
                x=torch.randn(1,64,24,32);pose=np.eye(4);pose[0,3]=i*.1
                meta=dict(timestamp_us=i*100000,segment='a',pose=pose)
                expected,states[0],_=base.step_spatial(x,states[0],meta)
                for j,m in enumerate(models,1):
                    actual,states[j],_=m.step_spatial(x,states[j],meta)
                    self.assertTrue(all(torch.equal(actual[k],v) for k,v in expected.items()))
                    self.assertTrue(all(torch.equal(states[j][k],states[0][k]) for k in base.state_keys))
        for h in handles:h.remove()
        self.assertEqual(calls,[])

    def test_true_skip_warp_and_unmodified_input_state(self):
        m=ReadoutRefreshDetector(fixture());calls=[];h=m.core.readout.register_forward_hook(lambda *_:calls.append(1))
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,s,_=m.step_spatial(x,None,meta);old=s['readout'].clone();pose=np.eye(4);pose[0,3]=.3
            bounds=block_lidar_range(m.bounds,x.shape[-2:],old.shape[-2:],2)
            aligned,valid=warp_bev(old,meta['pose'],pose,bounds)
            _,new,info=m.step_spatial(x,s,dict(meta,timestamp_us=100000,pose=pose))
            self.assertEqual(calls,[1]);self.assertFalse(info['readout_executed']);self.assertTrue(torch.equal(old,s['readout']))
            self.assertTrue(torch.equal(new['readout'].masked_select(valid),aligned.masked_select(valid)))
            self.assertEqual(info['cache_bytes'],sum(new[k].numel()*new[k].element_size() for k in (*m.state_keys,'readout')))
            for i in (2,3,4):_,new,info=m.step_spatial(x,new,dict(meta,timestamp_us=i*100000,pose=pose))
            self.assertEqual(calls,[1,1]);self.assertEqual(info['full_refresh'],'period')
        h.remove()

    def test_reset_age_and_causality(self):
        m=ReadoutRefreshDetector(fixture(),max_age=.15);x=torch.randn(1,64,24,32)
        meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,s,_=m.step_spatial(x,None,meta)
            _,_,info=m.step_spatial(x,s,dict(meta,timestamp_us=160000));self.assertEqual(info['full_refresh'],'age')
            for reason,extra in [('gap',dict(timestamp_us=300000)),('scene',dict(segment='b')),('explicit',dict(reset_before=True))]:
                _,r,info=m.step_spatial(x,s,dict(meta,**extra));self.assertEqual(info['reset'],reason)
                _,fresh,_=m.step_spatial(x,None,dict(meta,**extra));self.assertTrue(torch.equal(r['readout'],fresh['readout']))
            with self.assertRaises(ValueError):m.step_spatial(x,s,meta)

    def test_closed_loop_tbptt_gradients_only_fast(self):
        m=ReadoutRefreshDetector(fixture());s=None
        for n,p in m.named_parameters():p.requires_grad_(n.startswith('core.fast_readout.'))
        losses=[]
        for i in range(4):
            pose=np.eye(4);pose[0,3]=i*.2
            pred,s,_=m.step_spatial(torch.randn(1,64,24,32),s,dict(timestamp_us=i*100000,segment='a',pose=pose),detach_state=False)
            if i:losses.append(pred['psm'].square().mean())
        torch.stack(losses).mean().backward()
        self.assertIsNotNone(s['readout'].grad_fn)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad))
        opened=[p for p in m.parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in opened))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in opened),0)


if __name__=='__main__':unittest.main()

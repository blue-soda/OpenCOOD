import math
import unittest
import torch
from opencood.models.sub_modules.int_readout_amplitude import AmplitudeMemory,ReadoutAmplitudeAnchor,integer_delivery_statistics
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from test_int_pyramid_state import make_pair
from test_int_spatial_spike import meta


class AmplitudeTests(unittest.TestCase):
    def memory(self,mode,dtype=torch.float32):return AmplitudeMemory(mode,[1.],[0,0,-1,4,2,1],dtype)

    @torch.no_grad()
    def test_count_preserves_large_amplitude_without_saturation(self):
        m=self.memory('count');a=torch.tensor([0.,.1,.125,.3,1.,4.3,16.,100.]).reshape(1,1,2,4)
        y,state,info=m.step(a,None,meta(1))
        self.assertIsNone(state);self.assertEqual(int(m.last_counts.max()),800)
        self.assertTrue(torch.equal(y,.125*torch.floor(a/.125)))
        self.assertTrue(info['error_bound_checked']);self.assertGreater(float(y.max()),1.)

    @torch.no_grad()
    def test_residual_bound_and_reset_equals_stateless(self):
        torch.manual_seed(303);count=self.memory('count')
        for dtype in (torch.float16,torch.float32):
            m=self.memory('residual',dtype);state=None
            for i in range(40):
                a=torch.rand(1,1,2,4)*9;stamp=meta(1000000+i*100000)
                y,state,info=m.step(a,state,stamp);self.assertTrue(bool(((state['value']>=0)&(state['value']<=1)).all()))
                self.assertGreaterEqual(info['normalized_error_min'],-1.0001);self.assertLessEqual(info['normalized_error_max'],math.exp(-.5)+.0001)
                if i==39:
                    md=meta(1000000+(i+1)*100000);reset,_,_=m.step(a,state,md,reset=True);raw,_,_=count.step(a,None,md)
                    self.assertTrue(torch.equal(reset,raw));self.assertTrue(torch.equal(m.last_counts,count.last_counts))

    @torch.no_grad()
    def test_ema_has_unit_dc_gain_and_invalid_warp_uses_current(self):
        m=self.memory('ema');a=torch.full((1,1,2,4),3.5);state=None
        for i in range(30):
            y,state,_=m.step(a,state,meta(1000000+i*100000));self.assertTrue(torch.allclose(y,a,atol=1e-6))
        md=meta(4000000);md['pose'][0][3]=100
        current=torch.full_like(a,2.);y,_,info=m.step(current,state,md)
        self.assertEqual(info['valid_history_fraction'],0);self.assertTrue(torch.equal(y,current))

    @torch.no_grad()
    def test_scene_gap_causality_and_old_state_not_mutated(self):
        for mode in ('residual','ema'):
            m=self.memory(mode,torch.float16);a=torch.full((1,1,2,4),.8)
            _,state,_=m.step(a,None,meta(1000000));old=state['value'].clone()
            m.step(a,state,meta(1100000));self.assertTrue(torch.equal(old,state['value']))
            with self.assertRaises(ValueError):m.step(a,state,meta(1000000),reset=True)
            self.assertEqual(m.step(a,state,meta(1400000))[2]['reset'],'gap')
            self.assertEqual(m.step(a,state,meta(1,'other'))[2]['reset'],'scene')

    def test_frozen_only_and_overflow_rejection(self):
        with self.assertRaises(ValueError):self.memory('count').step(torch.ones(1,1,2,4),None,meta(1))
        with torch.no_grad():
            with self.assertRaises(RuntimeError):self.memory('count').step(torch.full((1,1,2,4),2.**24),None,meta(1))
            with self.assertRaises(RuntimeError):self.memory('ema',torch.float16).step(torch.full((1,1,2,4),70000.),None,meta(1))
            with self.assertRaises(ValueError):self.memory('count').step(-torch.ones(1,1,2,4),None,meta(1))

    def test_integer_event_multiplicity_and_bitplanes_match_brute_force(self):
        for h,w in ((1,1),(1,4),(2,4),(5,7)):
            counts=(torch.arange(h*w).reshape(1,1,h,w)*7+3)%33;stats=integer_delivery_statistics(counts,3)
            unary=bits=sparse=0
            for y in range(h):
                for x in range(w):
                    k=int(counts[0,0,y,x]);f=sum(0<=y+dy<h and 0<=x+dx<w for dy in (-1,0,1) for dx in (-1,0,1))
                    unary+=k*f*3;bits+=bin(k).count('1')*f*3;sparse+=int(k>0)*f*3
            self.assertEqual(unary,stats['unary_event_acs']);self.assertEqual(bits,stats['bitplane_additions']);self.assertEqual(sparse,stats['nonzero_sparse_macs'])
            self.assertEqual(sum(sum(v) for v in stats['count_histogram_by_fanout'].values()),h*w)

    @torch.no_grad()
    def test_wrapper_identity_reset_and_original_cache(self):
        torch.manual_seed(303);_,base=make_pair();strict=AnchoredLatePyramidDetector(base)
        models={n:ReadoutAmplitudeAnchor(base,n,torch.ones(256),torch.float16,n=='ema') for n in ('ann','count','residual','ema')}
        models['residual'].reset_activation_memory=True;states={n:None for n in models};gold=None
        for i in range(3):
            x=torch.rand(1,64,24,32);md=meta(1000000+i*100000);expected,gold,info=strict.step_spatial(x,gold,md);predictions={}
            for n,m in models.items():
                actual,states[n],di=m.step_spatial(x,states[n],md);predictions[n]=actual
                self.assertTrue(all(not p.requires_grad for p in m.parameters()))
                for k in ('value','coarse'):self.assertTrue(torch.equal(states[n][k],gold[k]))
                self.assertTrue(torch.equal(states[n]['anchor']['value'],gold['anchor']['value']))
                if n in ('ann','ema'):self.assertTrue(all(torch.equal(actual[k],v) for k,v in expected.items()))
                self.assertEqual(di['cache_bytes']-info['cache_bytes'],256*12*16*2 if n in ('residual','ema') else 0)
            self.assertTrue(all(torch.equal(predictions['count'][k],v) for k,v in predictions['residual'].items()))


if __name__=='__main__':unittest.main()

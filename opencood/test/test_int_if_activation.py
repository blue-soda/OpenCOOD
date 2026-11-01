import unittest
import numpy as np
import torch
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_if_activation import FixedPointIFActivation,coded_anchor,readout_coded_anchor
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector


class IFActivationTests(unittest.TestCase):
    def test_explicit_if_equals_integer_count_at_boundaries_and_random_inputs(self):
        torch.manual_seed(303)
        for steps in (4,8,16):
            x=torch.cat([torch.linspace(-.5,2.,127),torch.rand(137)*2,torch.tensor([0.,1./steps,1.,2.])]).reshape(1,1,1,-1).repeat(1,2,3,1)
            for scales in ([1.,1.],[.3,2.]):
                count=FixedPointIFActivation(scales,'count',steps).eval();spike=FixedPointIFActivation(scales,'if',steps).eval()
                with torch.no_grad():a=count(x);b=spike(x)
                self.assertTrue(torch.equal(a,b));self.assertEqual(count.last_stats['spike_count'],spike.last_stats['spike_count'])

    def test_clip_and_membrane_reset(self):
        clip=FixedPointIFActivation([1.],'clip').eval();spike=FixedPointIFActivation([1.],'if').eval()
        x=torch.tensor([-.2,0.,.2,1.,2.]).reshape(1,1,1,-1)
        with torch.no_grad():
            self.assertTrue(torch.equal(clip(x),x.clamp(0,1)))
            a=spike(x);spike(x*3);b=spike(x)
            self.assertTrue(torch.equal(a,b));self.assertEqual(spike.last_stats['persistent_membrane_bytes'],0)
            self.assertEqual(float(spike(-torch.ones_like(x)).sum()),0.)

    def test_shared_base_is_unchanged_and_reuse_skips_codec(self):
        base=fixture();model=coded_anchor(base,torch.ones(256),'if',8);codec=model.core.deep_update.residual[1]
        self.assertIsInstance(base.core.deep_update.residual[1],torch.nn.ReLU)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in model.parameters()))
        x=torch.randn(1,64,24,32);meta=dict(timestamp_us=0,segment='a',pose=np.eye(4))
        with torch.no_grad():
            _,state,_=model.step_spatial(x,None,meta);calls=codec.calls
            _,new,info=model.step_spatial(x,state,dict(meta,timestamp_us=100000))
            self.assertIsNone(info['late_refresh']);self.assertEqual(codec.calls,calls)
            self.assertIs(new['anchor']['value'],state['anchor']['value'])

    def test_invalid_configuration(self):
        for scale in ([0.],[-1.],[float('nan')]):
            with self.assertRaises(ValueError):FixedPointIFActivation(scale)
        with self.assertRaises(ValueError):FixedPointIFActivation([1.],steps=0)
        with self.assertRaises(ValueError):FixedPointIFActivation([1.],'zero',preserve_tail=True)

    def test_zero_intervention_is_exact(self):
        zero=FixedPointIFActivation([.1,1.],'zero').eval()
        with torch.no_grad():actual=zero(torch.randn(1,2,5,7)*10)
        self.assertEqual(int(torch.count_nonzero(actual)),0)
        self.assertEqual(zero.last_stats['if_neuron_updates'],0)

    def test_readout_codes_preserve_cache_on_refresh_and_reuse(self):
        base=fixture();scales=dict(input=torch.ones(384),middle=torch.ones(256))
        models=dict(ann=AnchoredLatePyramidDetector(base),count=readout_coded_anchor(base,scales,'count'),
                    spike=readout_coded_anchor(base,scales,'if'),zero=readout_coded_anchor(base,scales,'zero_mid'),
                    clip_input=readout_coded_anchor(base,scales,'clip_input'),clip_middle=readout_coded_anchor(base,scales,'clip_middle'),
                    clip_tail=readout_coded_anchor(base,scales,'clip',preserve_tail=True),
                    count_tail=readout_coded_anchor(base,scales,'count',preserve_tail=True),
                    if_tail=readout_coded_anchor(base,scales,'if',preserve_tail=True))
        self.assertIsInstance(base.core.readout.layers[0].double_conv[2],torch.nn.ReLU)
        states={k:None for k in models};x=torch.randn(1,64,24,32)
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models.values()}),1)
        with torch.no_grad():
            for frame in range(3):
                preds={};meta=dict(timestamp_us=frame*100000,segment='a',pose=np.eye(4))
                for name,model in models.items():
                    pred,states[name],_=model.step_spatial(x,states[name],meta);preds[name]=pred
                    for key in ('value','coarse'):self.assertTrue(torch.equal(states[name][key],states['ann'][key]))
                    self.assertTrue(torch.equal(states[name]['anchor']['value'],states['ann']['anchor']['value']))
                self.assertTrue(all(torch.equal(preds['count'][k],preds['spike'][k]) for k in preds['count']))
                self.assertTrue(all(torch.equal(preds['ann'][k],preds['clip_tail'][k]) for k in preds['ann']))
                self.assertTrue(all(torch.equal(preds['count_tail'][k],preds['if_tail'][k]) for k in preds['ann']))
                self.assertEqual(models['spike'].core.readout.middle_codec.calls,frame+1)

    def test_readout_rejects_signed_input(self):
        model=readout_coded_anchor(fixture(),dict(input=torch.ones(384),middle=torch.ones(256)))
        with torch.no_grad(),self.assertRaises(ValueError):model.core.readout(-torch.ones(1,384,3,3))

    def test_single_site_clipping_matches_manual_intervention(self):
        import copy
        base=fixture();scales=dict(input=torch.ones(384)*.1,middle=torch.ones(256)*.2)
        x=torch.rand(1,384,8,10)
        for mode in ('clip_input','clip_middle'):
            coded=readout_coded_anchor(base,scales,mode).core.readout
            reference=copy.deepcopy(base.core.readout)
            hooks=[]
            if mode=='clip_middle':hooks.append(reference.layers[0].double_conv[2].register_forward_hook(lambda m,args,y:y.clamp_max(.2)))
            with torch.no_grad():
                expected=reference(x.clamp_max(.1) if mode=='clip_input' else x)
                actual=coded(x)
            for hook in hooks:hook.remove()
            self.assertTrue(torch.equal(actual,expected))
            self.assertIsNone(coded.coding_stats()['middle' if mode=='clip_input' else 'input'])

    def test_tail_bypass_is_exact_and_exceptions_emit_no_spikes(self):
        x=torch.tensor([-.5,0.,.125,.5,1.,1.01,9.]).reshape(1,1,1,-1)
        for steps in (4,8,16):
            clip=FixedPointIFActivation([1.],'clip',steps,True).eval()
            count=FixedPointIFActivation([1.],'count',steps,True).eval()
            spike=FixedPointIFActivation([1.],'if',steps,True).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(clip(x),x.clamp_min(0)))
                a=count(x);b=spike(x)
            self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(a[x>1],x[x>1]))
            expected=int(torch.floor(x.clamp_min(0)[x<=1]*steps).sum())
            self.assertEqual(spike.last_stats['spike_count'],expected)
            self.assertEqual(spike.last_stats['analog_bypass_elements'],2)
            self.assertEqual(spike.last_stats['analog_value_payload_bytes'],8)
            self.assertEqual(spike.last_stats['if_neuron_updates'],steps*x.numel())

    def test_half_threshold_matches_nearest_count_and_resets(self):
        x=torch.tensor([0.,.0625,.125,.1875,.99,1.,1.01,9.]).reshape(1,1,1,-1)
        count=FixedPointIFActivation([1.],'count',8,True,32768).eval()
        spike=FixedPointIFActivation([1.],'if',8,True,32768).eval()
        expected=torch.where(x>1,x,torch.floor(torch.floor(x*65536).clamp_max(65536)*8/65536+.5)/8)
        with torch.no_grad():
            a=count(x);b=spike(x);spike(x*3);again=spike(x)
        self.assertTrue(torch.equal(a,expected))
        self.assertTrue(torch.equal(a,b) and torch.equal(a,again))
        self.assertEqual(float(a.flatten()[0]),0.)
        with self.assertRaises(ValueError):FixedPointIFActivation([1.],initial_membrane_units=1)


if __name__=='__main__':unittest.main()

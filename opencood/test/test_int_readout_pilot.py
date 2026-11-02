import unittest
from unittest import mock
import torch
from opencood.tools.pilot_int_readout_spike import (TARGETS,scaled_backward,
    supervised_updates,frozen_digest,evaluation_arms,readout_capture)
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from test_int_pyramid_state import make_pair
from test_int_spatial_spike import meta


class ReadoutPilotTests(unittest.TestCase):
    def test_supervised_windows_respect_context_and_boundaries(self):
        rows=[dict(frame=str(i),timestamp_us=100000*i,segment='a' if i<6 else 'b',supervised=i in (1,5,8)) for i in range(10)]
        self.assertEqual(supervised_updates(rows),['3','5','9'])
        rows[5]['supervised']=False
        self.assertEqual(supervised_updates(rows),['3','9'])

    def test_unscale_before_clip_and_reject_nonfinite(self):
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Parameter(torch.tensor([3.]));self.b=torch.nn.Parameter(torch.tensor([4.]));self.frozen=torch.nn.Parameter(torch.tensor([2.]),requires_grad=False)
        with mock.patch('opencood.tools.pilot_int_readout_spike.TARGETS',('a','b')):
            m=Fixture();result=scaled_backward(m,m.a.square().sum()+m.b.square().sum())
            self.assertAlmostEqual(result['gradient_norm_before_clip'],10.)
            self.assertTrue(torch.allclose(m.a.grad,torch.tensor([6.])));self.assertTrue(torch.allclose(m.b.grad,torch.tensor([8.])))
            self.assertIsNone(m.frozen.grad)
            m.zero_grad(set_to_none=True)
            with self.assertRaises(RuntimeError):scaled_backward(m,m.a.sum()*float('inf'))

    def test_frozen_digest_excludes_only_four_parameters(self):
        _,base=make_pair();m=ReadoutSpikeAnchor(base,'ann',torch.ones(256));before=frozen_digest(m)
        with torch.no_grad():
            for n,p in m.named_parameters():
                if n in TARGETS:p.add_(.01)
        self.assertEqual(before,frozen_digest(m))
        m.core.readout.layers[0].double_conv[1].running_mean.add_(.01)
        self.assertNotEqual(before,frozen_digest(m))

    def test_precision_control_copies_weights_and_never_aliases_membrane(self):
        torch.manual_seed(303);_,base=make_pair();strict=AnchoredLatePyramidDetector(base)
        models={n:ReadoutSpikeAnchor(base,n,torch.ones(256),torch.float16) for n in ('ann','lif','leaky')}
        arms=evaluation_arms(strict,models)
        for n in ('lif','leaky'):
            self.assertIs(arms[n],arms[n+'_reset']);self.assertIsNot(arms[n],arms[n+'_fp32'])
            self.assertEqual(arms[n+'_fp32'].activation_memory.storage_dtype,torch.float32)
            self.assertEqual(arms[n].activation_memory.storage_dtype,torch.float16)
            for key,p in arms[n].named_parameters():
                q=dict(arms[n+'_fp32'].named_parameters())[key]
                self.assertTrue(torch.equal(p,q));self.assertNotEqual(p.data_ptr(),q.data_ptr())
        with torch.no_grad(),readout_capture(arms['lif']) as observed:
            _,state,_=arms['lif'].step_spatial(torch.rand(1,64,24,32),None,meta(1000000))
            self.assertEqual(observed['common'].shape[1],384);self.assertEqual(observed['output'].shape[1],256)
            self.assertEqual(state['readout_membrane']['value'].dtype,torch.float16)
        self.assertFalse(any(m._forward_pre_hooks or m._forward_hooks for m in arms['lif'].modules()))


if __name__=='__main__':unittest.main()

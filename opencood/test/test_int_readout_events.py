import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_readout_events import ChangeEncodedReadout,ReadoutEventDetector


class ReadoutEventsTests(unittest.TestCase):
    def test_static_input_no_events_and_refresh_exact(self):
        source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval()
        model=ChangeEncodedReadout(source,[[.125]*2,[.125]*3]);x=torch.randn(1,2,4,5)
        with torch.no_grad():
            initial=model(x);self.assertTrue(torch.equal(initial,source(x)))
            model.previous=model.next_state;model.refresh=False;again=model(x)
            self.assertTrue(torch.equal(initial,again));self.assertEqual(sum(r['unit_spike_ac'] for r in model.statistics['layers']),0)
            model.refresh=True;changed=x+1;self.assertTrue(torch.equal(model(changed),source(changed)))

    def test_signed_events_boundary_fanout_and_linear_identity(self):
        source=nn.Sequential(nn.Conv2d(2,3,3,padding=1)).eval();model=ChangeEncodedReadout(source,[[.25]*2]);x=torch.zeros(1,2,3,4)
        with torch.no_grad():
            model(x);old=model.next_state;copy=old[0].clone();model.previous=old;model.refresh=False
            x[0,0,0,0]=.5;x[0,1,1,1]=-.75;x[0,0,2,3]=.1;y=model(x);q=model.next_state[0];r=model.statistics['layers'][0]
            self.assertTrue(torch.equal(old[0],copy));self.assertEqual(r['nonzero_events'],2)
            self.assertEqual(r['weighted_event_mac'],(4+9)*3);self.assertEqual(r['unit_spike_ac'],(2*4+3*9)*3)
            self.assertTrue(torch.allclose(y-source(old[0]),F.conv2d(q-old[0],source[0].weight,padding=1),atol=1e-6,rtol=1e-6))
            self.assertLessEqual(r['quantization_error_in_quanta'],1.)

    def test_invalid_quanta_and_gradients_rejected(self):
        source=nn.Sequential(nn.Conv2d(2,3,1))
        for q in ([0,1],[float('nan'),1],[1]):
            with self.assertRaises(ValueError):ChangeEncodedReadout(source,[q])
        model=ChangeEncodedReadout(source,[[.1,.1]])
        with self.assertRaises(ValueError):model(torch.zeros(1,2,2,2))

    def test_detector_reset_upstream_and_external_execution_state(self):
        source=ProtectedMultiscaleRefreshDetector(fixture()).eval();quanta=[[.125]*m.in_channels for m in source.core.readout.modules() if isinstance(m,nn.Conv2d)];model=ReadoutEventDetector(source,quanta)
        a=b=None;x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for i in range(6):
                meta=dict(timestamp_us=i*100000,segment='a' if i<5 else 'b',pose=np.eye(4));pa,a,ia=source.step_spatial(x,a,meta);pb,b,ib=model.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in source.state_keys))
                self.assertEqual(ia['full_refresh'],ib['full_refresh'])
                if ia['full_refresh']:self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa))
                self.assertEqual(ib['cache_bytes']-ia['cache_bytes'],sum(t.numel()*4 for t in b['readout']))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import torch
from torch import nn
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_compact_causal_integer_state import narrow_integer_state,CompactIntegerReadoutDetector


class CompactIntegerStateTests(unittest.TestCase):
    def test_signed_boundaries_and_lossless_fallback(self):
        for reference in (False,True):
            for values in ([-32768,32767],[-32769,32768],[-2**31,2**31-1],[-2**31-1,2**31]):
                x=torch.tensor(values,dtype=torch.int64);y=narrow_integer_state(x,reference);self.assertTrue(torch.equal(x,y.long()))
                expected=torch.int16 if reference and values==[-32768,32767] else torch.int64 if values==[-2**31-1,2**31] else torch.int32
                self.assertEqual(y.dtype,expected)

    def test_own_state_closed_loop_preserves_integer_predictions(self):
        torch.manual_seed(303);source=ProtectedMultiscaleRefreshDetector(fixture()).eval();q=[[.125]*m.in_channels for m in source.core.readout.modules() if isinstance(m,nn.Conv2d)];dense=IntegerReadoutDetector(source,q,False);compact=CompactIntegerReadoutDetector(source,q,True);a=b=None
        with torch.no_grad():
            for i in range(7):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=i*100000,segment='a' if i<5 else 'b',pose=np.eye(4));pa,a,ia=dense.step_spatial(x,a,meta);pb,b,ib=compact.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa));self.assertTrue(all(torch.equal(v,w.long()) for key in ('readout','readout_accumulators') for v,w in zip(a[key],b[key])))
                self.assertLess(ib['cache_bytes'],ia['cache_bytes'])


if __name__=='__main__':unittest.main()

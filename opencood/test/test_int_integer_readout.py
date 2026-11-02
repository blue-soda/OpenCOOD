import unittest
import numpy as np
import torch
from torch import nn
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_integer_readout import IntegerReadout,IntegerReadoutDetector


class IntegerReadoutTests(unittest.TestCase):
    def test_independent_nonlinear_stream_bit_exact_and_reset(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).eval();quanta=[[.125]*2,[.125]*3]
        dense=IntegerReadout(source,quanta,False);sparse=IntegerReadout(source,quanta,True)
        with torch.no_grad():
            for i in range(8):
                dense.refresh=sparse.refresh=i in (0,4);x=torch.randn(1,2,4,5);a=dense(x);b=sparse(x);self.assertTrue(torch.equal(a,b))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(dense.next_state,sparse.next_state)));self.assertTrue(all(torch.equal(a,b) for a,b in zip(dense.next_accumulators,sparse.next_accumulators)))
                self.assertEqual(sum(v['dense_reference_calls'] for v in sparse.statistics['layers']),2 if sparse.refresh else 0)
                for m in (dense,sparse):m.previous=m.next_state;m.previous_accumulators=m.next_accumulators

    def test_detector_external_integer_state_and_unchanged_task_memory(self):
        torch.manual_seed(303);source=ProtectedMultiscaleRefreshDetector(fixture()).eval();q=[[.125]*m.in_channels for m in source.core.readout.modules() if isinstance(m,nn.Conv2d)];dense=IntegerReadoutDetector(source,q,False);sparse=IntegerReadoutDetector(source,q,True);a=b=plain=None
        self.assertEqual(sum(p.numel() for p in dense.parameters()),sum(p.numel() for p in source.parameters()))
        with torch.no_grad():
            for i in range(6):
                x=torch.randn(1,64,24,32);meta=dict(timestamp_us=i*100000,segment='a' if i<5 else 'b',pose=np.eye(4));_,plain,ip=source.step_spatial(x,plain,meta);pa,a,ia=dense.step_spatial(x,a,meta);pb,b,ib=sparse.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa));self.assertTrue(all(torch.equal(a[k],plain[k]) and torch.equal(b[k],plain[k]) for k in source.state_keys))
                self.assertEqual(ia['cache_bytes'],ib['cache_bytes']);self.assertEqual(ia['cache_bytes']-ip['cache_bytes'],sum(t.numel()*8 for t in a['readout']+a['readout_accumulators']))
                self.assertTrue(all(t.dtype==torch.int64 for t in b['readout']+b['readout_accumulators']))


if __name__=='__main__':unittest.main()

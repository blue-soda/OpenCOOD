import unittest
import numpy as np
import torch
from torch import nn
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_readout_events import ChangeEncodedReadout,ReadoutEventDetector
from opencood.models.sub_modules.int_sparse_readout_events import SparseChangeEncodedReadout,SparseReadoutEventDetector


class SparseReadoutTests(unittest.TestCase):
    def test_causal_nonlinear_stream_and_no_dense_calls(self):
        torch.manual_seed(303);source=nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,2,1)).double().eval();quanta=[[.25]*2,[.25]*3]
        dense=ChangeEncodedReadout(source,quanta);sparse=SparseChangeEncodedReadout(source,quanta);calls=[]
        handles=[m.register_forward_hook(lambda *_:calls.append(1)) for m in sparse.layers if isinstance(m,nn.Conv2d)]
        with torch.no_grad():
            for i in range(8):
                refresh=i in (0,4);dense.refresh=sparse.refresh=refresh;x=torch.randn(1,2,4,5,dtype=torch.float64);calls.clear();a=dense(x);b=sparse(x)
                self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12));self.assertEqual(len(calls),2 if refresh else 0)
                dense.previous=dense.next_state;sparse.previous=sparse.next_state;sparse.previous_accumulators=sparse.next_accumulators
                if not refresh:self.assertEqual(sum(v['executed_macs'] for v in sparse.execution),sum(v['weighted_event_mac'] for v in sparse.statistics['layers']))
        for h in handles:h.remove()

    def test_external_accumulators_reset_and_parameter_identity(self):
        torch.manual_seed(303);source=ProtectedMultiscaleRefreshDetector(fixture()).eval();quanta=[[.125]*m.in_channels for m in source.core.readout.modules() if isinstance(m,nn.Conv2d)];dense=ReadoutEventDetector(source,quanta);sparse=SparseReadoutEventDetector(source,quanta)
        self.assertEqual(sum(p.numel() for p in dense.parameters()),sum(p.numel() for p in sparse.parameters()));a=b=None;x=torch.randn(1,64,24,32)
        with torch.no_grad():
            for i in range(6):
                meta=dict(timestamp_us=i*100000,segment='a' if i<5 else 'b',pose=np.eye(4));pa,a,ia=dense.step_spatial(x,a,meta);pb,b,ib=sparse.step_spatial(x,b,meta)
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in source.state_keys));self.assertTrue(all(torch.allclose(pa[k],pb[k],atol=5e-4,rtol=1e-4) for k in pa))
                self.assertEqual(ib['cache_bytes']-ia['cache_bytes'],sum(t.numel()*4 for t in b['readout_accumulators']))
                if ia['full_refresh']:self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa))


if __name__=='__main__':unittest.main()

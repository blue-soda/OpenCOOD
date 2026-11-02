import unittest
import torch
from torch import nn
from opencood.models.sub_modules.resblock import BasicBlock
from opencood.models.sub_modules.int_fine_integer import FineIntegerEncoder
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadout
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout


class FineReadoutIntegerTests(unittest.TestCase):
    def test_composed_nonlinear_own_state_and_distinct_refresh(self):
        torch.manual_seed(303)
        source=nn.Sequential(BasicBlock(2,4,2,nn.Sequential(nn.Conv2d(2,4,1,stride=2,bias=False),nn.BatchNorm2d(4))),BasicBlock(4,4),BasicBlock(4,4)).eval()
        calibration={n:dict(quanta=[.125]*c.in_channels) for n,c in source.named_modules() if isinstance(c,nn.Conv2d)}
        fine=[FineIntegerEncoder(source,calibration,m).eval() for m in ('dense','hybrid')]
        reader=nn.Sequential(nn.Conv2d(4,3,3,padding=1),nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU(),nn.Conv2d(4,3,3,padding=1),nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU()).eval()
        quanta=[[.125]*c for c in (4,3,4,3)]
        readout=[IntegerReadout(reader,quanta,False).eval(),ImplicitReferenceIntegerReadout(reader,quanta,True).eval()]
        fs=[{},{}];rs=[None,None];acc=[None,None]
        with torch.no_grad():
            for step in range(9):
                x=torch.randn(1,2,7,8);outputs=[];features=[]
                for i,(f,r) in enumerate(zip(fine,readout)):
                    for n,leaf in f.leaves().items():leaf.previous=None if step in (0,5) else fs[i].get(n)
                    z=f(x);features.append(z);fs[i]={n:l.next_state for n,l in f.leaves().items() if l.next_state is not None}
                    r.previous=rs[i];r.previous_accumulators=acc[i];r.refresh=step in (0,3,5,8)
                    outputs.append(r(z));rs[i]=r.next_state;acc[i]=r.next_accumulators
                self.assertTrue(torch.equal(features[0],features[1]));self.assertTrue(torch.equal(outputs[0],outputs[1]))
                self.assertTrue(torch.equal(rs[0][0],rs[1][0]))
                self.assertTrue(all(torch.equal(a,b.to(a.dtype)) for a,b in zip(acc[0],acc[1])))
                self.assertEqual(len(rs[1]),1)


if __name__=='__main__':unittest.main()

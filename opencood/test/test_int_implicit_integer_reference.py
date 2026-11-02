import copy,unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadout
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout


class ImplicitReferenceTests(unittest.TestCase):
    def run_stream(self,scale):
        torch.manual_seed(303)
        source=nn.Sequential(nn.Conv2d(3,4,3,padding=1,bias=True),nn.Conv2d(4,5,1,bias=True),nn.BatchNorm2d(5),nn.ReLU(),nn.Conv2d(5,4,3,padding=1,bias=True),nn.Conv2d(4,3,1,bias=True),nn.BatchNorm2d(3),nn.ReLU()).eval()
        quanta=[[.125]*c for c in (3,4,5,4)]
        reference=CompactIntegerReadout(source,quanta,True);implicit=ImplicitReferenceIntegerReadout(source,quanta,True)
        states=[None,None];accs=[None,None];saw_wide=False
        with torch.no_grad():
            for t in range(8):
                x=torch.randn(1,3,5,7)*scale;refresh=t in (0,4)
                if t==6:x.zero_()
                if states[1] is not None and not refresh:
                    restored=implicit.reconstruct_references(states[1],accs[1])
                    self.assertTrue(all(torch.equal(a,b.to(a.dtype)) for a,b in zip(restored,states[0])))
                values=[]
                for i,m in enumerate((reference,implicit)):
                    m.previous=states[i];m.previous_accumulators=accs[i];m.refresh=refresh
                    values.append(m(x));states[i]=m.next_state;accs[i]=m.next_accumulators
                self.assertTrue(torch.equal(values[0],values[1]))
                self.assertTrue(all(torch.equal(a,b.to(a.dtype)) for a,b in zip(accs[0],accs[1])))
                self.assertTrue(torch.equal(states[0][0],states[1][0]))
                self.assertEqual(len(states[1]),1)
                self.assertLess(implicit.statistics['sending_reference_bytes'],reference.statistics['sending_reference_bytes'])
                self.assertEqual([z['actual_integer_macs'] for z in implicit.statistics['layers']],[z['actual_integer_macs'] for z in reference.statistics['layers']])
                saw_wide|=states[1][0].dtype!=torch.int16 or any(a.dtype==torch.int64 for a in accs[1])
        return saw_wide

    def test_transforms_recurrence_and_refresh(self):
        self.run_stream(1.)

    def test_wide_state_fallback_remains_exact(self):
        self.assertTrue(self.run_stream(100000.))


if __name__=='__main__':unittest.main()

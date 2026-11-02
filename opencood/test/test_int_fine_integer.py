import unittest
import torch
from torch import nn
from opencood.models.sub_modules.resblock import BasicBlock
from opencood.models.sub_modules.int_fine_integer import FineIntegerEncoder, FineIntegerConv, FINE_POLICY


class FineIntegerTests(unittest.TestCase):
    def stream(self, magnitude):
        torch.manual_seed(303)
        source=nn.Sequential(BasicBlock(2,4,2,nn.Sequential(nn.Conv2d(2,4,1,stride=2,bias=False),nn.BatchNorm2d(4))),BasicBlock(4,4),BasicBlock(4,4)).eval()
        calibration={n:dict(quanta=[.125]*c.in_channels) for n,c in source.named_modules() if isinstance(c,nn.Conv2d)}
        models=[FineIntegerEncoder(source,calibration,m).eval() for m in ('dense','static','hybrid')]
        states=[{}, {}, {}]; wide=False
        with torch.no_grad():
            for step in range(7):
                x=torch.randn(1,2,7,8)*magnitude
                if step==5:x.zero_()
                outputs=[]
                for i,m in enumerate(models):
                    for n,leaf in m.leaves().items():
                        leaf.previous=None if step in (0,3) else states[i].get(n);leaf.capture_audit=True
                    outputs.append(m(x));states[i]={n:l.next_state for n,l in m.leaves().items() if l.next_state is not None}
                self.assertTrue(torch.equal(outputs[0],outputs[1]));self.assertTrue(torch.equal(outputs[0],outputs[2]))
                self.assertEqual(states[0],{});self.assertEqual(states[1],{})
                self.assertEqual(set(states[2]),{n for n,v in FINE_POLICY.items() if v=='delta'})
                for n,reference in models[0].leaves().items():
                    for m in models[1:]:
                        leaf=m.leaves()[n];self.assertTrue(all(torch.equal(a,b) for a,b in zip(reference.audit,leaf.audit)))
                        self.assertEqual(leaf.statistics['actual_integer_macs'],leaf.statistics['event_mac'])
                        self.assertLessEqual(leaf.statistics['quantization_error_in_quanta'],.500001)
                wide|=any(n.dtype!=torch.int16 or a.dtype==torch.int64 for n,a in states[2].values())
        return wide

    def test_three_block_own_state_and_rebuild(self): self.stream(1.)

    def test_wide_storage_fallback(self): self.assertTrue(self.stream(100000.))

    def test_calibration_and_inference_guards(self):
        conv=nn.Conv2d(2,3,3,stride=2,padding=1)
        with self.assertRaises(ValueError):FineIntegerConv(conv,[0.,.1],'static')
        leaf=FineIntegerConv(conv,[.1,.1],'static').eval()
        with self.assertRaises(ValueError):leaf(torch.ones(1,2,5,6))


if __name__=='__main__':unittest.main()

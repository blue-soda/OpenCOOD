import io,unittest
import torch
from torch import nn
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerReadout


class StaticRefreshReadoutTests(unittest.TestCase):
    def run_stream(self,scale):
        torch.manual_seed(303)
        source=nn.Sequential(nn.Conv2d(3,4,3,padding=1,bias=True),nn.Conv2d(4,5,1,bias=True),nn.BatchNorm2d(5),nn.ReLU(),nn.Conv2d(5,4,3,padding=1,bias=True),nn.Conv2d(4,3,1,bias=True),nn.BatchNorm2d(3),nn.ReLU()).eval()
        quanta=[[.125]*c for c in (3,4,5,4)]
        models=[ImplicitReferenceIntegerReadout(source,quanta,True),ImplicitReferenceIntegerReadout(source,quanta,True,sparse_refresh=True)]
        states=[None,None];accs=[None,None];wide=False
        with torch.no_grad():
            for t in range(9):
                x=torch.randn(1,3,5,7)*scale;refresh=t in (0,4,7)
                if t in (3,7):x.zero_()
                saved=(states[1],accs[1]);outputs=[]
                for i,m in enumerate(models):
                    m.previous=states[i];m.previous_accumulators=accs[i];m.refresh=refresh
                    outputs.append(m(x));states[i]=m.next_state;accs[i]=m.next_accumulators
                self.assertTrue(torch.equal(outputs[0],outputs[1]))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(states[0],states[1])))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(accs[0],accs[1])))
                for old,new in zip(models[0].statistics['layers'],models[1].statistics['layers']):
                    self.assertEqual(new['actual_fp64_conv_flops'],0)
                    self.assertEqual(new['dense_reference_calls'],0)
                    self.assertEqual(new['actual_integer_macs'],new['weighted_event_mac'])
                    self.assertLessEqual(new['actual_integer_macs'],new['dense_mac'])
                    if not refresh:self.assertEqual(old,new)
                for key in ('sending_reference_bytes','accumulator_bytes','reference_dtypes','accumulator_dtypes','stored_reference_count'):
                    self.assertEqual(models[0].statistics[key],models[1].statistics[key])
                if t==5:
                    stream=io.BytesIO();torch.save(saved,stream);stream.seek(0);restored=torch.load(stream)
                    m=models[1];m.previous,m.previous_accumulators=restored;m.refresh=False
                    self.assertTrue(torch.equal(m(x),outputs[1]))
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.next_accumulators,accs[1])))
                wide|=states[1][0].dtype!=torch.int16 or any(a.dtype==torch.int64 for a in accs[1])
        return wide

    def test_static_refresh_and_recurrence_and_restore(self):self.run_stream(1.)
    def test_wide_integer_states(self):self.assertTrue(self.run_stream(100000.))
    def test_sparse_mode_required(self):
        with self.assertRaises(ValueError):ImplicitReferenceIntegerReadout(nn.Sequential(nn.Conv2d(2,2,1)),[[.125,.125]],False,sparse_refresh=True)


if __name__=='__main__':unittest.main()

import io,unittest
import numpy as np
import torch
from test_int_combined_integer import refresh_fixture
from opencood.models.sub_modules.int_combined_integer import CombinedIntegerDetector
from opencood.models.sub_modules.int_composed_static_integer import ComposedStaticIntegerDetector


class ComposedStaticIntegerTests(unittest.TestCase):
    def test_full_causal_stream_exact_state_storage_and_restore(self):
        source,fine,reader,candidate,encoder=refresh_fixture()
        models=[CombinedIntegerDetector(source,fine,reader,candidate,encoder,'static','static',True),
                ComposedStaticIntegerDetector(source,fine,reader,candidate,encoder,True)]
        states=[None,None]
        schedule=[(i*100000,'a',False) for i in range(6)]+[(600000,'b',False),(700000,'b',False),(1100000,'b',False),(1200000,'b',True)]
        with torch.no_grad():
            for step,(stamp,scene,explicit) in enumerate(schedule):
                x=torch.randn(1,64,8,8);pose=np.eye(4);pose[0,3]=step*.05
                meta=dict(timestamp_us=stamp,segment=scene,pose=pose,reset_before=explicit)
                before=states[1];preds=[];infos=[]
                for i,m in enumerate(models):
                    pred,states[i],info=m.step_spatial(x,states[i],meta);preds.append(pred);infos.append(info)
                self.assertTrue(all(torch.equal(preds[0][k],preds[1][k]) for k in preds[0]))
                self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in source.state_keys))
                for key in ('readout','readout_accumulators'):
                    self.assertTrue(all(torch.equal(a,b) and a.dtype==b.dtype for a,b in zip(states[0][key],states[1][key])))
                for key in ('fine_integer','candidate_integer','encoder_integer','cache_bytes','audit_scratch_bytes','full_refresh','reset'):
                    self.assertEqual(infos[0][key],infos[1][key])
                for i in (0,1):
                    self.assertEqual(states[i]['candidate_integer'],{});self.assertEqual(states[i]['encoder_integer'],{})
                for name,leaf in models[0].core.observation.leaves().items():
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(leaf.audit,models[1].core.observation.leaves()[name].audit)))
                for group in ('candidate','encoder'):
                    for name,leaf in getattr(models[0],group+'_leaves')().items():
                        other=getattr(models[1],group+'_leaves')()[name]
                        self.assertEqual(leaf.audit is None,other.audit is None)
                        if leaf.audit is not None:self.assertTrue(all(torch.equal(a,b) for a,b in zip(leaf.audit,other.audit)))
                self.assertTrue(all(z['actual_fp64_conv_flops']==z['dense_reference_calls']==0 for z in infos[1]['readout_events']['layers']))
                if not infos[0]['full_refresh']:self.assertEqual(infos[0]['readout_events'],infos[1]['readout_events'])
                if step in (4,5):
                    stream=io.BytesIO();torch.save(before,stream);stream.seek(0)
                    pred,new,info=models[1].step_spatial(x,torch.load(stream),meta)
                    self.assertTrue(all(torch.equal(pred[k],preds[1][k]) for k in pred))
                    self.assertTrue(all(torch.equal(new[k],states[1][k]) for k in source.state_keys))
                    self.assertEqual(info,infos[1])
                for m in models:m.clear_audit()


if __name__=='__main__':unittest.main()

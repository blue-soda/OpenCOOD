"""A repeated output drift must be preserved, never silently accepted."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from opencood.tools.strong_ann_baselines.compute_harness import fingerprint
from opencood.tools.strong_ann_baselines.diagnose_snapshot_compute import diagnose


class Fake:
    def prepare_cpu(self,index):return 'frame',index
    def reset(self,fid):pass
    def to_device(self,batch):return batch
    def infer(self,batch):return batch
    def to_cpu(self,result):return dict(boxes=np.zeros((1,8,3),np.float32),scores=np.asarray([.51],np.float32))
    def packets(self):return [('road_to_vehicle',b'unchanged')]


class DiagnosticTests(unittest.TestCase):
    def test_preserves_score_only_drift_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'ref';out=root/'out';(ref/'predictions').mkdir(parents=True);out.mkdir()
            arrays=dict(boxes=np.zeros((1,8,3),np.float32),scores=np.asarray([.5],np.float32))
            np.savez_compressed(str(ref/'predictions/f.npz'),pred_boxes=arrays['boxes'],scores=arrays['scores'])
            (ref/'predictions/index.json').write_text(json.dumps(dict(frames=[dict(file='f.npz')])))
            refs={'frame':fingerprint('frame',arrays,[('road_to_vehicle',b'unchanged')])}
            result=diagnose(Fake(),[0],refs,ref,out)
            self.assertEqual(result['mismatched_calls'],2)
            self.assertEqual(result['packet_mismatched_calls'],0)
            self.assertEqual(result['unstable_repeat_calls'],0)
            self.assertEqual(result['rows'][0]['changed_fields'],['scores'])
            self.assertGreater(result['rows'][0]['scores_max_abs'],0)
            self.assertTrue((out/'mismatch_r1_i0000/actual.npz').exists())
            self.assertTrue(result['reference_gate_not_relaxed'])


if __name__=='__main__':unittest.main()

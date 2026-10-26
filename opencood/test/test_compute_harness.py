"""CPU-only regression of timing boundary, message identity and warmup scope."""
import unittest
import numpy as np
from opencood.tools.strong_ann_baselines.compute_harness import (
    selected_indices, fingerprint, measure_once, run_pass)


class FakeAdapter:
    def __init__(self):
        self.events=[];self.fid=None;self.payload=b'actual-packet'
    def prepare_cpu(self,index):
        self.events.append('prepare');return str(index),index
    def reset(self,fid):self.events.append('reset');self.fid=fid
    def to_device(self,batch):self.events.append('h2d');return batch
    def infer(self,batch):self.events.append('infer');return batch
    def to_cpu(self,result):
        self.events.append('d2h');return {'boxes':np.zeros((1,8,3),np.float32),'scores':np.ones(1,np.float32)}
    def packets(self):self.events.append('packet_extract');return [('road_to_vehicle',self.payload)]
    def sync(self):self.events.append('sync')
    def clock(self):self.events.append('clock');return len(self.events)*.1
    def reference(self,fid):
        return fingerprint(fid,{'boxes':np.zeros((1,8,3),np.float32),'scores':np.ones(1,np.float32)},[('road_to_vehicle',b'actual-packet')])


class ComputeHarnessTests(unittest.TestCase):
    def test_boundaries_and_hashes_outside_timing(self):
        for scope in ['ready_cpu_batch_to_cpu_boxes','device_batch_to_cpu_boxes']:
            a=FakeAdapter()
            row=measure_once(a,0,'0',a.reference('0'),scope=scope,synchronize=a.sync,clock=a.clock)
            first=a.events.index('clock');last=len(a.events)-1-a.events[::-1].index('clock')
            self.assertLess(a.events.index('infer'),last)
            self.assertLess(a.events.index('d2h'),last)
            self.assertGreater(a.events.index('packet_extract'),last)
            self.assertEqual(a.events.index('h2d')>first,scope=='ready_cpu_batch_to_cpu_boxes')
            self.assertTrue(row['reference_exact'])

    def test_wrong_bytes_same_length_and_wrong_frame_fail(self):
        for payload,fid in [(b'changed-byte!', '0'),(b'actual-packet','1')]:
            a=FakeAdapter();a.payload=payload
            with self.assertRaises(ValueError):
                measure_once(a,0,fid,a.reference('0'),scope='ready_cpu_batch_to_cpu_boxes',synchronize=a.sync,clock=a.clock)

    def test_packet_labels_and_nonfinite_predictions_fail(self):
        arrays={'boxes':np.zeros((1,8,3),np.float32),'scores':np.ones(1,np.float32)}
        with self.assertRaises(ValueError):fingerprint('0',arrays,[('same',b'a'),('same',b'b')])
        arrays['scores'][0]=np.nan
        with self.assertRaises(ValueError):fingerprint('0',arrays,[])

    def test_warmup_not_measured_and_resource_data_not_filtered(self):
        a=FakeAdapter();refs={str(i):a.reference(str(i)) for i in [0,3]};snapshots=[]
        def resource():snapshots.append(1);return {'other_gpu_processes':[123]}
        result=run_pass(a,[0,3],refs,scope='ready_cpu_batch_to_cpu_boxes',synchronize=a.sync,
                        resource_snapshot=resource,warmup=3,repeat=2,clock=a.clock)
        self.assertEqual(a.events.count('infer'),5)
        self.assertEqual([x['index'] for x in result['rows']],[0,3])
        self.assertEqual(len(snapshots),2)
        self.assertEqual(result['resource_after']['other_gpu_processes'],[123])
        self.assertFalse(result['interference_filter_applied'])

    def test_fixed_sample_contract(self):
        indices=selected_indices()
        self.assertEqual((indices[0],indices[-1],len(set(indices))),(0,1737,64))
        with self.assertRaises(ValueError):selected_indices(3,4)


if __name__=='__main__':unittest.main()

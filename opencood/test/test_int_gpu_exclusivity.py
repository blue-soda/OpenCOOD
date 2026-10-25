import unittest
from unittest.mock import patch
from opencood.tools.int_gpu_exclusivity import verify_gpu_processes,check_gpu_exclusive,GPUContention


class TestGpuExclusivity(unittest.TestCase):
    def test_other_device_allowed_same_device_rejected(self):
        devices='0, GPU-a\n1, GPU-b\n'
        self.assertEqual(verify_gpu_processes(devices,'GPU-a, 101\nGPU-b, 202',1,202)['pids'],[202])
        with self.assertRaisesRegex(RuntimeError,'contention'):
            verify_gpu_processes(devices,'GPU-b, 202\nGPU-b, 303',1,202)
        with self.assertRaisesRegex(RuntimeError,'absent'):
            verify_gpu_processes(devices,'GPU-a, 101',1,202)

    def query(self,processes):
        replies=[]
        for rows in processes:replies.extend([b'1, GPU-b\n',rows.encode()])
        return patch('opencood.tools.int_gpu_exclusivity.subprocess.check_output',side_effect=replies)

    @patch('opencood.tools.int_gpu_exclusivity.os.getpid',return_value=202)
    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    def test_clean_does_not_wait(self,sleep,pid):
        with self.query(['GPU-b, 202']):result=check_gpu_exclusive(1)
        self.assertTrue(result['timing_valid']);sleep.assert_not_called()

    @patch('opencood.tools.int_gpu_exclusivity.os.getpid',return_value=202)
    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.process_identity',side_effect=lambda p:dict(pid=p,Uid='17'))
    def test_transient_keeps_evidence_and_invalidates_timing(self,identity,sleep,pid):
        with self.query(['GPU-b, 202\nGPU-b, 303','GPU-b, 202']):result=check_gpu_exclusive(1)
        self.assertFalse(result['timing_valid'])
        self.assertEqual(result['transient_contention'][0]['processes'],[dict(pid=303,Uid='17')])
        sleep.assert_called_once_with(1.)

    @patch('opencood.tools.int_gpu_exclusivity.os.getpid',return_value=202)
    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.process_identity',side_effect=lambda p:dict(pid=p))
    def test_persistent_exits_with_both_samples(self,identity,sleep,pid):
        with self.query(['GPU-b, 202\nGPU-b, 303','GPU-b, 202\nGPU-b, 404']):
            with self.assertRaises(GPUContention) as caught:check_gpu_exclusive(1)
        self.assertEqual(caught.exception.pids,[404])
        self.assertEqual([x['processes'][0]['pid'] for x in caught.exception.evidence],[303,404])
        sleep.assert_called_once_with(1.)

import unittest
from unittest.mock import patch
from opencood.tools.int_gpu_exclusivity import verify_gpu_processes,check_gpu_exclusive,GPUContention,wait_for_training_gpu


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
        with self.query(['GPU-b, 202']) as query:result=check_gpu_exclusive(1)
        self.assertTrue(result['timing_valid']);sleep.assert_not_called()
        for call in query.call_args_list:self.assertEqual(call[0][0][:3],['nvidia-smi','-i','1'])

    @patch('opencood.tools.int_gpu_exclusivity.os.getpid',return_value=202)
    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.process_identity',side_effect=lambda p:dict(pid=p,Uid='17'))
    def test_transient_keeps_evidence_and_invalidates_timing(self,identity,sleep,pid):
        with self.query(['GPU-b, 202\nGPU-b, 303','GPU-b, 202']):result=check_gpu_exclusive(1)
        self.assertFalse(result['timing_valid'])
        self.assertEqual(result['transient_contention'][0]['processes'],[dict(pid=303,Uid='17')])
        sleep.assert_called_once_with(1.)

    def test_device_errors_are_not_silently_treated_as_free(self):
        with self.assertRaisesRegex(RuntimeError,'device query unavailable'):
            verify_gpu_processes('Unable to determine device handle','GPU-b, 202',1,202)
        with self.assertRaisesRegex(RuntimeError,'process query unavailable'):
            verify_gpu_processes('1, GPU-b','Unknown Error',1,202)

    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.time.monotonic',side_effect=[0.,1.,16.])
    def test_training_pauses_then_resumes_with_evidence(self,clock,sleep):
        contention=GPUContention(1,[303]);contention.evidence=[dict(pid=303)]
        events=[]
        with patch('opencood.tools.int_gpu_exclusivity.check_gpu_exclusive',side_effect=[contention,dict(timing_valid=True)]):
            result=wait_for_training_gpu(1,on_event=events.append)
        self.assertEqual([e['status'] for e in events],['WAITING_GPU','RESUMED'])
        self.assertFalse(result['timing_valid']);self.assertEqual(result['training_wait_seconds'],16.)
        sleep.assert_called_once_with(15)

    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.time.monotonic',side_effect=[0.,2.])
    def test_training_wait_timeout_and_hardware_fail_closed(self,clock,sleep):
        events=[]
        with patch('opencood.tools.int_gpu_exclusivity.check_gpu_exclusive',side_effect=GPUContention(1,[303])):
            with self.assertRaises(GPUContention):wait_for_training_gpu(1,timeout_seconds=1,on_event=events.append)
        self.assertEqual(events[-1]['status'],'WAIT_TIMEOUT');sleep.assert_not_called()
        with patch('opencood.tools.int_gpu_exclusivity.time.monotonic',return_value=0.),patch('opencood.tools.int_gpu_exclusivity.check_gpu_exclusive',side_effect=RuntimeError('device unavailable')):
            with self.assertRaisesRegex(RuntimeError,'device unavailable'):wait_for_training_gpu(1)

    @patch('opencood.tools.int_gpu_exclusivity.os.getpid',return_value=202)
    @patch('opencood.tools.int_gpu_exclusivity.time.sleep')
    @patch('opencood.tools.int_gpu_exclusivity.process_identity',side_effect=lambda p:dict(pid=p))
    def test_persistent_exits_with_both_samples(self,identity,sleep,pid):
        with self.query(['GPU-b, 202\nGPU-b, 303','GPU-b, 202\nGPU-b, 404']):
            with self.assertRaises(GPUContention) as caught:check_gpu_exclusive(1)
        self.assertEqual(caught.exception.pids,[404])
        self.assertEqual([x['processes'][0]['pid'] for x in caught.exception.evidence],[303,404])
        sleep.assert_called_once_with(1.)

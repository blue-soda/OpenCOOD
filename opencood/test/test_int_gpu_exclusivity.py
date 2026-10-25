import unittest
from opencood.tools.int_gpu_exclusivity import verify_gpu_processes


class TestGpuExclusivity(unittest.TestCase):
    def test_other_device_allowed_same_device_rejected(self):
        devices='0, GPU-a\n1, GPU-b\n'
        self.assertEqual(verify_gpu_processes(devices,'GPU-a, 101\nGPU-b, 202',1,202)['pids'],[202])
        with self.assertRaisesRegex(RuntimeError,'contention'):
            verify_gpu_processes(devices,'GPU-b, 202\nGPU-b, 303',1,202)
        with self.assertRaisesRegex(RuntimeError,'absent'):
            verify_gpu_processes(devices,'GPU-a, 101',1,202)

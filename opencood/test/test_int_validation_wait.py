"""Validation pauses preserve ordered state updates and invalidate latency."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from opencood.tools import train_int_recurrent_deep as runner
from opencood.tools.int_gpu_exclusivity import GPUContention


class Trajectory:
    readout_cache = None

    def __init__(self):
        self.seen = []

    def step(self, ego, state, meta):
        current = 0 if state is None else state + 1
        assert current == meta['index']
        self.seen.append(current)
        return {'rm': torch.zeros(1)}, current, dict(deep_partial=True,
            local_partial=True, spatial_refresh=True, state_bytes=4,
            spatial_state_bytes=0, anchor_bytes=0)


class TestValidationWait(unittest.TestCase):
    def evaluate(self, path, baseline, student, wait, shared=False):
        rows = [dict(supervised=False, index=i) for i in range(12)]
        samples = [dict(ego={}, meta=r) for r in rows]
        with patch.object(runner, 'loader', return_value=(None, samples)), \
             patch.object(runner, 'to_device', side_effect=lambda x, d: x), \
             patch.object(runner.torch.cuda, 'synchronize'), \
             patch.object(runner.eval_utils, 'calculate_ap', return_value=(.7553159775482435,)):
            return runner.evaluate(SimpleNamespace(gpu=1), baseline, None, {}, rows,
                path, {('dense', True): 10}, 0, 14,
                variants={'student': student}, wait_for_gpu=wait, share_gpu=shared)

    def test_persistent_sharing_completes_ap_without_wait_or_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'epoch_014';directory.mkdir()
            baseline,student=Trajectory(),Trajectory()
            with patch.object(runner,'sample_gpu_usage',return_value=dict(timing_valid=False,foreign_pids=[99])), \
                 patch.object(runner,'wait_for_training_gpu') as wait, \
                 patch.object(runner,'check_gpu_exclusive') as strict:
                report=self.evaluate(directory,baseline,student,False,shared=True)
            wait.assert_not_called();strict.assert_not_called()
            self.assertEqual(baseline.seen,list(range(12)))
            self.assertEqual(student.seen,baseline.seen)
            self.assertFalse(report['timing_valid'])
            self.assertEqual(report['status'],'COMPLETED')
            for result in report['results'].values():
                self.assertIsNone(result['mean_ms']);self.assertIsNone(result['p95_ms'])
                self.assertEqual(result['ap70'],.7553159775482435)

    def test_pause_keeps_every_frame_once_and_nulls_whole_run_latency(self):
        events = []

        def wait(index, on_event):
            events.append(index)
            if len(events) == 1:
                on_event(dict(status='WAITING_GPU', evidence=[dict(pid=99)]))
                on_event(dict(status='RESUMED'))
                return dict(timing_valid=False, training_contention=[dict(pid=99)])
            return dict(timing_valid=True)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'epoch_014'; directory.mkdir()
            baseline, student = Trajectory(), Trajectory()
            with patch.object(runner, 'wait_for_training_gpu', side_effect=wait), \
                 patch.object(runner, 'check_gpu_exclusive') as strict:
                report = self.evaluate(directory, baseline, student, True)
            strict.assert_not_called()
            self.assertEqual(baseline.seen, list(range(12)))
            self.assertEqual(student.seen, baseline.seen)
            self.assertEqual(events, [1, 1])
            self.assertFalse(report['timing_valid'])
            self.assertEqual(report['scans'], 12)
            for result in report['results'].values():
                self.assertIsNone(result['mean_ms']); self.assertIsNone(result['p95_ms'])
                self.assertEqual(result['ap70'], .7553159775482435)
            self.assertEqual(json.loads((directory / 'gpu_exclusivity.json').read_text())['status'], 'INVALID_TIMING')

    def test_strict_default_and_wait_timeout_both_fail_before_next_frame(self):
        for wait in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / 'epoch_014'; directory.mkdir()
                baseline, student = Trajectory(), Trajectory()
                method = 'wait_for_training_gpu' if wait else 'check_gpu_exclusive'
                with patch.object(runner, method, side_effect=GPUContention(1, [99])):
                    with self.assertRaises(GPUContention):
                        self.evaluate(directory, baseline, student, wait)
                self.assertEqual(baseline.seen, [])
                self.assertEqual(student.seen, [])
                self.assertFalse((directory / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()

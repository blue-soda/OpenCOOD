import unittest
import torch
from opencood.models.sub_modules.int_event_training_loss import (
    READOUT_DENSE_MAC, window_objective, gradient_summary)


class EventTrainingLossTests(unittest.TestCase):
    def test_all_frames_including_refresh_are_denominator(self):
        detection = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        units = torch.tensor(float(READOUT_DENSE_MAC), dtype=torch.float64, requires_grad=True)
        total, det, event = window_objective([detection], [units * 0, units], True)
        self.assertEqual(float(det), 2.)
        self.assertAlmostEqual(float(event), .05)
        self.assertAlmostEqual(float(total), 2.005)
        total.backward()
        self.assertEqual(float(detection.grad), 1.)
        self.assertAlmostEqual(float(units.grad) * READOUT_DENSE_MAC, .005)

    def test_zero_coefficient_keeps_original_detection_graph(self):
        x = torch.tensor(2., requires_grad=True)
        detached_reference = torch.tensor(5., requires_grad=True)
        total, detection, _ = window_objective([x.square()], [detached_reference], False)
        self.assertIs(total, detection)
        total.backward()
        self.assertEqual(float(x.grad), 4.)
        self.assertIsNone(detached_reference.grad)

    def test_detection_mean_uses_only_labels(self):
        total, detection, _ = window_objective([torch.tensor(2.), torch.tensor(6.)],
                                              [torch.tensor(0.)] * 4, True)
        self.assertEqual(float(detection), 4.)
        self.assertEqual(float(total), 4.)
        with self.assertRaises(ValueError):window_objective([], [torch.tensor(1.)], True)
        with self.assertRaises(ValueError):window_objective([torch.tensor(1.)], [], True)

    def test_gradient_coverage_does_not_hide_disconnected_parameters(self):
        s = gradient_summary(['a', 'b', 'c'], [torch.tensor([3., 4.]), None, torch.zeros(2)])
        self.assertTrue(s['finite']);self.assertEqual(s['norm'], 5.)
        self.assertEqual(s['nonzero_tensors'], 1)
        self.assertFalse(s['parameters']['b']['present'])
        self.assertFalse(gradient_summary(['bad'], [torch.tensor(float('nan'))])['finite'])


if __name__ == '__main__':unittest.main()

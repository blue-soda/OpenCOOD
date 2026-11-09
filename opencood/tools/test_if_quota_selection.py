"""Regression checks against the reference event stream, including address ties."""
import unittest

import torch

from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import (
    select_event_quota, select_if_window, select_tensor_quota)


class IFQuotaSelectionTest(unittest.TestCase):
    def compare(self, values, device):
        drive = values.reshape(1, 1, 1, 1, -1).to(device).requires_grad_()
        reference = integrate_and_fire(drive, 8)
        grad_output = torch.arange(1, drive.numel() + 1, device=device).reshape_as(drive)
        for quota in (0, 1, 2, 3, 7, 20, len(reference.times), len(reference.times) + 1, None):
            expected = select_event_quota(reference, quota).counts
            actual = select_tensor_quota(drive, quota).counts
            self.assertTrue(torch.equal(actual, expected))
            grads = [torch.autograd.grad(y, drive, grad_outputs=grad_output,
                                         retain_graph=True)[0] for y in (expected, actual)]
            self.assertTrue(torch.equal(grads[0], grads[1]))
        for window in (0., 0.125, 0.25, 0.373, 1.):
            n = int((reference.times <= window).sum())
            self.assertTrue(torch.equal(select_if_window(drive, window).counts, reference.prefix(n)))

    def test_reference(self):
        torch.manual_seed(19)
        cases = [torch.tensor([8., 4., 2., 9., 0., 1.]), torch.zeros(17),
                 torch.full((64,), 8.), torch.rand(501) * 9]
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for values in cases:
                with self.subTest(device=device, size=len(values)):
                    self.compare(values, device)


if __name__ == '__main__':
    unittest.main()

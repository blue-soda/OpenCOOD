import unittest
import torch
from opencood.utils.e3dsnn_aux_loss import detection_loss


class AuxLossTest(unittest.TestCase):
    def test_weights_and_gradients(self):
        for weight in (0., 1.):
            fused = torch.tensor([2.], requires_grad=True)
            ego = torch.tensor([3.], requires_grad=True)
            output = {k: fused for k in ('psm', 'rm', 'dm')}
            output['ego_output'] = {k: ego for k in ('psm', 'rm', 'dm')}
            loss, parts = detection_loss(output, {}, lambda o, _: o['psm'].square().sum(), weight)
            self.assertEqual(float(loss), 4 + weight * 9)
            loss.backward()
            self.assertEqual(float(fused.grad), 4.)
            self.assertEqual(float(ego.grad), weight * 6.)
            self.assertEqual(parts['ego_detection_loss'], 9.)

    def test_missing_aux_and_invalid_aux_fail(self):
        output = {k: torch.tensor([1.]) for k in ('psm', 'rm', 'dm')}
        criterion = lambda o, _: o['psm'].sum()
        with self.assertRaises(ValueError): detection_loss(output, {}, criterion, 1.)
        output['ego_output'] = {k: torch.tensor([float('nan')]) for k in ('psm', 'rm', 'dm')}
        with self.assertRaises(ValueError): detection_loss(output, {}, criterion, 1.)


if __name__ == '__main__':
    unittest.main()

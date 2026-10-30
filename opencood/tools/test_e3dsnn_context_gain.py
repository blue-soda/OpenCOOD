"""CPU tests of identity, condition isolation and staged gradient admission."""
import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_context_gain import SnapshotContextGain


class ContextGainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260929)
        self.local = torch.rand(1, 128, 2, 3, 4)
        self.remote = torch.rand_like(self.local)

    def test_identity_and_matched_capacity(self):
        a, b = SnapshotContextGain('local'), SnapshotContextGain('message')
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in a.parameters()), 6288)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        for gate in (a, b):
            self.assertTrue(torch.equal(gate(self.local, self.remote), torch.ones_like(self.local)))
            self.assertTrue(torch.equal(self.remote * gate(self.local, self.remote), self.remote))

    def test_condition_isolation(self):
        a, b = SnapshotContextGain('local'), SnapshotContextGain('message')
        with torch.no_grad():
            a.net[0].weight.fill_(.001); a.net[0].bias.fill_(.1)
            a.net[-1].weight.fill_(.01)
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertTrue(torch.equal(a(self.local, self.remote), a(self.local, self.remote + 1)))
        self.assertFalse(torch.equal(b(self.local, self.remote), b(self.local, self.remote + 1)))

    def test_two_step_gradient(self):
        for mode in ('local', 'message'):
            gate = SnapshotContextGain(mode)
            optimizer = torch.optim.SGD(gate.parameters(), lr=.1)
            for step in range(2):
                optimizer.zero_grad()
                (gate(self.local, self.remote) * self.remote).square().mean().backward()
                self.assertGreater(float(gate.net[-1].weight.grad.abs().sum()), 0)
                hidden_grad = float(gate.net[0].weight.grad.abs().sum())
                if step == 0:
                    self.assertEqual(hidden_grad, 0)
                else:
                    self.assertGreater(hidden_grad, 0)
                optimizer.step()

    def test_reject_invalid_condition_and_shape(self):
        with self.assertRaises(ValueError): SnapshotContextGain('unknown')
        with self.assertRaises(ValueError): SnapshotContextGain('local')(self.local, self.remote[:, :1])


if __name__ == '__main__':
    unittest.main()

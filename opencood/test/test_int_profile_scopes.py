import unittest
from unittest.mock import patch
import torch
from opencood.tools.profile_int_basis_regions import EventScopes


class FakeEvent:
    def __init__(self, **kwargs): pass
    def record(self): pass
    def elapsed_time(self, other): return 1.


class ProfileScopeTests(unittest.TestCase):
    def test_registered_module_identity_and_restore(self):
        model = torch.nn.Module()
        model.current = torch.nn.Linear(3, 2).eval()
        original = model.current
        weight = original.weight
        x = torch.ones(1, 3)
        expected = original(x)
        scopes = EventScopes()
        with patch('torch.cuda.Event', FakeEvent):
            try:
                scopes.wrap(model, 'current', 'readout')
                self.assertIs(model.current, original)
                self.assertIs(model.current.weight, weight)
                self.assertTrue(torch.equal(model.current(x), expected))
                self.assertEqual(scopes.values()['readout'][2], 1)
            finally:
                scopes.close()
        before = len(scopes.events)
        self.assertTrue(torch.equal(model.current(x), expected))
        self.assertEqual(len(scopes.events), before)


if __name__ == '__main__': unittest.main()

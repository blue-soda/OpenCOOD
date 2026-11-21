import unittest
from types import SimpleNamespace

import torch

from opencood.tools.eval_int_silent_middle import SilencePolicy


class SilentFootprintTest(unittest.TestCase):
    def test_asymmetric_grid_halo_and_history(self):
        events = torch.zeros(1,1,13,17,dtype=torch.bool)
        events[...,0,0] = events[...,9,12] = True
        skip = torch.zeros(1,1,7,9,dtype=torch.bool)
        skip[...,0,8] = True
        valid = torch.ones_like(skip)
        valid[...,6,0] = False
        coverage = torch.ones_like(skip)
        coverage[...,0,7] = False
        expected = torch.zeros_like(skip)
        allowed = ~skip & valid & coverage
        for y in range(7):
            for x in range(9):
                patch = events[...,max(0,2*y-5):2*y+6,max(0,2*x-5):2*x+6]
                expected[...,y,x] = allowed[...,y,x] & ~patch.any()
        self.assertGreater(int(expected.sum()),0)
        policies = [SilencePolicy(SimpleNamespace(event_sites=events),n) for n in ('silent','continuous')]
        for p in policies:
            chosen = p(skip,valid,coverage,skip.float(),1)
            self.assertEqual(int(p.extra.sum()),int(expected.sum()))
            self.assertFalse(bool((p.extra & ~allowed).any()))
            self.assertTrue(torch.equal(chosen,skip|p.extra))
        self.assertTrue(torch.equal(policies[0].extra,expected))

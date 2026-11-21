import itertools
import unittest

import torch

from opencood.models.sub_modules.int_middle_trace_receiver import pack_directions, unpack_directions


class PackedDirectionTest(unittest.TestCase):
    def test_all_four_channel_patterns_are_distinct_and_reversible(self):
        values=torch.tensor(list(itertools.product((-1.,0.,1.),repeat=4))).T
        wire=pack_directions(values)
        self.assertEqual(wire.shape,(1,81))
        self.assertEqual(wire.unique().numel(),81)
        self.assertTrue(torch.equal(unpack_directions(wire),values))

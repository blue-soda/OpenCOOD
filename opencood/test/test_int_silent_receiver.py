import unittest

import torch

from opencood.models.sub_modules.int_silent_receiver import SilentReceiver


class ThresholdTest(unittest.TestCase):
    def run_trace(self, forget, targets, caps):
        r = SilentReceiver(6, forget=forget)
        zero = torch.zeros(1, 1, 1, 1)
        r.refresh(zero)
        seen = []
        for frame, (value, cap) in enumerate(zip(targets, caps)):
            c = r.prepare(zero + value*.25, torch.tensor([0]), lambda x: x.clone())
            out, s = r.receive(c, cap, frame, zero.bool().fill_(True), [3, 6, 13, 28])
            seen.append((s['addresses'], int(out.item()*4), s['silent_tiles']))
        return seen

    def test_gradual_subthreshold_changes(self):
        for sign in (-1, 1):
            targets = [sign*x for x in (2, 4, 6)]
            self.assertEqual(self.run_trace(False, targets, [1]*3), [(0,0,1),(0,0,1),(1,6*sign,0)])
            self.assertEqual(self.run_trace(True, targets, [1]*3), [(0,0,1)]*3)

    def test_threshold_crossing_survives_denied_grant(self):
        for forget in (True, False):
            self.assertEqual(self.run_trace(forget, [9, 9], [0,1]), [(0,0,1),(1,6,0)])

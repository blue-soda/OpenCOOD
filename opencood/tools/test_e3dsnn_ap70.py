"""AP70 matching and dual-checkpoint regression checks, CPU only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from opencood.tools.diagnose_e3dsnn_ap70 import diagnose, match
from opencood.tools.test_e3dsnn_prediction_replay import PredictionReplayTest
from opencood.tools.train_e3dsnn_single import save_best_checkpoints


class DiagnosticTest(PredictionReplayTest):
    def test_identical_bundles_and_empty_frames(self):
        result = diagnose(self.folder, self.folder, expected_frames=3)
        self.assertTrue(result['passed'])
        self.assertEqual(result['gt_transitions70'],
                         dict(retained=1, lost=0, gained=0, missed_by_both=1))
        self.assertEqual(result['models']['ego']['counts']['fp_best_iou_at_least_07'], 1)

    def test_score_order_and_remaining_gt(self):
        order, tp, gt = match(np.asarray([[.9, .8], [.95, .1], [.1, .1]]),
                              np.asarray([.8, .9, .7]), .7)
        self.assertEqual(order.tolist(), [1, 0, 2])
        self.assertEqual(tp.tolist(), [1, 1, 0])
        self.assertEqual(gt, [0, 1, -1])


class CheckpointTest(unittest.TestCase):
    def test_ap70_selection_preserves_ap50_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model = torch.nn.Linear(1, 1, bias=False)
            best_by = {'0.5': -1., '0.7': -1.}
            best = -1.
            for epoch, ap50, ap70 in [(1, .8, .6), (2, .9, .5), (3, .85, .6), (4, .86, .7)]:
                model.weight.data.fill_(epoch)
                best = save_best_checkpoints(model, dict(epoch=epoch, ap={'0.5': ap50, '0.7': ap70}),
                                             folder, best_by, '0.7', best)
                if epoch == 3:
                    self.assertEqual(json.loads((folder/'best_metrics.json').read_text())['epoch'], 1)
            self.assertEqual(best_by, {'0.5': .9, '0.7': .7})
            for filename, epoch in [('best_ap50.pth', 2), ('best_ap70.pth', 4), ('best.pth', 4)]:
                self.assertEqual(torch.load(str(folder/filename), map_location='cpu')['weight'].item(), epoch)


if __name__ == '__main__':
    unittest.main()

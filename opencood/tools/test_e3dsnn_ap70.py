"""AP70 matching and dual-checkpoint regression checks, CPU only."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from opencood.tools.diagnose_e3dsnn_ap70 import diagnose, match, bev_error, pr_summary
from opencood.tools.test_e3dsnn_prediction_replay import PredictionReplayTest
from opencood.tools.train_e3dsnn_single import save_best_checkpoints
from opencood.tools.launch_e3dsnn_ap70_controls import launch


class DiagnosticTest(PredictionReplayTest):
    def test_geometry_translation_and_heading_wrap(self):
        box = np.asarray([[-2, -1, 0], [2, -1, 0], [2, 1, 0], [-2, 1, 0]] * 2, dtype=float)
        moved = box + [0.3, 0.4, 0]
        self.assertAlmostEqual(bev_error(moved, box)['center_error_m'], .5)
        flipped = box.copy()
        flipped[:, :2] *= -1
        self.assertAlmostEqual(bev_error(flipped, box)['axis_error_deg'], 0.)
        turned = box.copy()
        turned[:, 0], turned[:, 1] = -box[:, 1], box[:, 0]
        self.assertAlmostEqual(bev_error(turned, box)['axis_error_deg'], 90.)

    def test_pr_area_split_and_unreached_recall(self):
        report = pr_summary(dict(tp=[1, 0], fp=[0, 1], score=[.9, .8], gt=2))
        self.assertAlmostEqual(report['ap70'], .5)
        self.assertAlmostEqual(report['ap70_by_recall_interval']['0.0:0.3'], .3)
        self.assertAlmostEqual(report['ap70_by_recall_interval']['0.3:0.5'], .2)
        self.assertEqual(report['precision_envelope_at_recall']['0.6'], 0.)

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


class LaunchGuardTest(unittest.TestCase):
    def test_busy_gpu_and_existing_launch_never_start_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('count4', 'relu'):
                (root/(name+'_diagnostic.json')).write_text(json.dumps(dict(passed=True)))
            checkpoint = root/'weights.pth'
            checkpoint.write_bytes(b'fixture')
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            jobs = [dict(command=['--checkpoint', str(checkpoint)]), {},
                    dict(command=['--single-checkpoint', str(checkpoint)])]
            (root/'count4_controls_plan.json').write_text(json.dumps(dict(name='count4_controls',
                single_checkpoint_sha256=digest, fusion_checkpoint_sha256=digest, jobs=jobs)))
            with patch('opencood.tools.launch_e3dsnn_ap70_controls.subprocess.check_output', return_value=b'22000, 2000, 0'), \
                 patch('opencood.tools.launch_e3dsnn_ap70_controls.subprocess.Popen') as spawn:
                self.assertFalse(launch(root, 'count4', 0))
                spawn.assert_not_called()
                self.assertFalse((root/'count4_controls_launch.json').exists())
                (root/'count4_controls_launch.json').write_text('{}')
                with self.assertRaises(FileExistsError):
                    launch(root, 'count4', 0)
                spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()

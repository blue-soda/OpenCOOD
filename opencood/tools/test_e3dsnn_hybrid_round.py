import unittest
from pathlib import Path
from opencood.tools.launch_e3dsnn_hybrid_round import make_jobs


class HybridPlanTest(unittest.TestCase):
    def test_zero_does_not_train_and_uses_composed_weights(self):
        name, jobs = make_jobs('/experiment', 'sa', 'zero')
        self.assertEqual(len(jobs), 2)
        cmd = jobs[0]['command']
        self.assertEqual(cmd[cmd.index('--mode') + 1], 'eval')
        self.assertEqual(cmd[cmd.index('--checkpoint') + 1], str(Path('/experiment/sa_initial.pth')))
        self.assertIn('--packet-roundtrip', cmd)

    def test_adaptation_matches_budget_and_keeps_off_receiver(self):
        outputs = set()
        for combo in ('ss', 'sa', 'as', 'aa'):
            name, jobs = make_jobs('/experiment', combo, 'adapt10')
            self.assertEqual(len(jobs), 6)
            train = jobs[0]['command']
            self.assertEqual(train[train.index('--epochs') + 1], '10')
            self.assertNotIn('--warmup-epochs', train)
            self.assertEqual(train[train.index('--selection-iou') + 1], '0.7')
            outputs.add(train[train.index('--output') + 1])
            off = jobs[3]['command']
            self.assertTrue(off[off.index('--config') + 1].endswith('e3dsnn_hybrid_' + combo + '_ego_181.yaml'))
            self.assertNotIn('--packet-roundtrip', off)
        self.assertEqual(len(outputs), 4)


if __name__ == '__main__':
    unittest.main()

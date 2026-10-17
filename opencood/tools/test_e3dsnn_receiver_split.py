import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from opencood.utils.e3dsnn_receiver_split import compose_receiver_state
from opencood.tools.launch_e3dsnn_receiver_round import make_jobs, source_gate, factorial_contrasts


class ReceiverSplitTest(unittest.TestCase):
    def donors(self):
        keys = ['backbone_3d.weight', 'backbone_3d.running_mean', 'fusion_scale',
                'fusion_net.weight', 'backbone_2d.running_var', 'reg_head.weight',
                'cls_head.weight', 'dir_head.weight']
        s = {k: torch.tensor([1.]) for k in keys}
        a = {k: torch.tensor([1. if k.startswith('backbone_3d.') else 2.]) for k in keys}
        return s, a

    def test_all_partitions_and_native_identity(self):
        s, a = self.donors()
        for fusion, bev in [('s', 's'), ('a', 's'), ('s', 'a'), ('a', 'a')]:
            state = compose_receiver_state(s, a, fusion, bev)
            for k, v in state.items():
                expected = s if k.startswith('backbone_3d.') else (s if
                    (fusion if k == 'fusion_scale' or k.startswith('fusion_net.') else bev) == 's' else a)
                self.assertTrue(torch.equal(v, expected[k]))
            state['reg_head.weight'].add_(100)
            self.assertEqual(float(s['reg_head.weight']), 1.)
            self.assertEqual(float(a['reg_head.weight']), 2.)

    def test_reject_encoder_drift_and_unknown_tensors(self):
        s, a = self.donors()
        for key in ['backbone_3d.weight', 'backbone_3d.running_mean']:
            bad = copy.deepcopy(a); bad[key].add_(1)
            with self.assertRaises(ValueError): compose_receiver_state(s, bad, 'a', 's')
        s['unknown'] = a['unknown'] = torch.ones(1)
        with self.assertRaises(ValueError): compose_receiver_state(s, a, 's', 'a')

    def test_budget_and_path_isolation(self):
        for combo in ['ss', 'as', 'sa', 'aa']:
            name, jobs = make_jobs('/receiver', combo, 'adapt10')
            self.assertEqual(name, 'receiver_' + combo + '_adapt10')
            self.assertEqual(len(jobs), 6)
            train = jobs[0]['command']
            self.assertEqual(train[train.index('--epochs') + 1], '10')
            self.assertEqual(train[train.index('--checkpoint') + 1], str(Path('/receiver') / (combo + '_initial.pth')))
            self.assertNotIn('--warmup-epochs', train)
            self.assertIn('e3dsnn_receiver_' + combo + '_ego_181.yaml', ' '.join(jobs[3]['command']))
            self.assertNotIn('e3dsnn_hybrid_', ' '.join(x for job in jobs for x in job['command']))
            _, zero = make_jobs('/receiver', combo, 'zero')
            self.assertEqual(len(zero), 2)

    def test_source_cannot_start_before_previous_round_completes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'round_status.json').write_text(json.dumps(dict(state='running')))
            with self.assertRaises(AssertionError): source_gate(path)

    def test_factor_order_and_percentage_point_units(self):
        values = factorial_contrasts({'ss': .50, 'as': .51, 'sa': .54, 'aa': .56})
        self.assertAlmostEqual(values['ann_fusion_with_snn_bev'], 1.)
        self.assertAlmostEqual(values['ann_fusion_with_ann_bev'], 2.)
        self.assertAlmostEqual(values['ann_bev_with_snn_fusion'], 4.)
        self.assertAlmostEqual(values['ann_bev_with_ann_fusion'], 5.)
        self.assertAlmostEqual(values['interaction'], 1.)


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
import torch
from opencood.utils.e3dsnn_budget import budget_features
from opencood.utils.quantized_feature_packet import prepare


class BudgetTest(unittest.TestCase):
    def test_forward_is_wire_and_backward_is_selected_only(self):
        indices = torch.tensor([[b, 0, 0, x] for b in (0, 1) for x in range(5)], dtype=torch.int32)
        for activation, bits in [('count4', 3), ('relu', 2)]:
            raw = np.random.RandomState(6).uniform(0, 4, (10, 128)).astype(np.float32)
            if activation == 'count4': raw = np.floor(raw)
            values = torch.tensor(raw, requires_grad=True)
            options = dict(bits=bits, keep_fraction=.5)
            actual = budget_features(indices, values, activation, options)
            expected = np.zeros_like(raw); kept = []
            for b in (0, 1):
                c, q, scale = prepare(indices[b*5:b*5+5, 1:].numpy(), raw[b*5:b*5+5], activation, options)
                rows = b*5 + c[:, 2].astype(int); kept.extend(rows)
                expected[rows] = q.astype(np.float32)*scale
            np.testing.assert_array_equal(actual.detach().numpy(), expected)
            actual.sum().backward()
            expected_grad = np.zeros_like(raw); expected_grad[kept] = 1
            np.testing.assert_array_equal(values.grad.numpy(), expected_grad)

    def test_empty_zero_and_permutation(self):
        indices = torch.tensor([[0, 0, 0, x] for x in range(3)], dtype=torch.int32)
        values = torch.zeros(3, 128, requires_grad=True)
        options = dict(bits=3, keep_fraction=.25)
        actual = budget_features(indices, values, 'count4', options)
        actual.sum().backward(); self.assertEqual(float(values.grad.abs().sum()), 0)
        values = torch.ones(3, 128)
        a = budget_features(indices, values, 'count4', options)
        b = budget_features(indices.flip(0), values.flip(0), 'count4', options).flip(0)
        self.assertTrue(torch.equal(a, b))

    def test_fixed_budget_pipeline_contract(self):
        from pathlib import Path
        from opencood.tools.launch_e3dsnn_budget_round import GROUPS, make_jobs
        self.assertEqual(len(GROUPS), 4)
        root = Path('/audit/budget')
        for group in GROUPS:
            name, jobs = make_jobs(root, group, 'adapt10')
            self.assertEqual(name, 'budget_'+group+'_adapt10')
            train = jobs[0]['command']
            self.assertEqual(train[train.index('--epochs')+1], '10')
            self.assertEqual(train[train.index('--config')+1], str(root/(group+'.yaml')))
            self.assertEqual(jobs[3]['command'][jobs[3]['command'].index('--config')+1], str(root/(group+'_ego.yaml')))
            self.assertEqual(len(make_jobs(root,group,'zero')[1]),2)


if __name__ == '__main__': unittest.main()

import unittest
import numpy as np
from opencood.tools.test_e3dsnn_event_stream import metadata
from opencood.tools.profile_e3dsnn_event_granularity import encode_snapshot
from opencood.utils.snapshot_selection import encode_rows, select_snapshot


class SelectionTest(unittest.TestCase):
    def test_sparse_codec_identity_and_validation(self):
        m = metadata(); rng = np.random.RandomState(41)
        for _ in range(20):
            a = rng.randint(0, 9, (1, 32, 1, 1, 4)).astype(np.uint8)
            a[rng.rand(*a.shape) < .6] = 0
            rows = a.reshape(32, 4).T
            coords = np.flatnonzero(np.any(rows != 0, axis=1))
            self.assertEqual(encode_snapshot(a, m), encode_rows(coords, rows[coords], m))
        with self.assertRaises(ValueError): encode_rows([2, 1], np.ones((2, 32)), m)
        with self.assertRaises(ValueError): encode_rows([0], np.full((1, 32), 9), m)

    def test_every_budget_matches_brute_force_and_row_grouping(self):
        m = metadata(); ids = np.array([0, 5, 3, 0, 9, 6, 3, 8, 5], dtype=np.uint32)
        full = np.bincount(ids, minlength=128).reshape(1, 32, 1, 1, 4)
        for mode in ('events', 'rows'):
            boundaries = [0, len(encode_snapshot(full, m))+1]
            candidates = []; a = np.zeros_like(full)
            order = list(range(len(ids))) if mode == 'events' else [0, 1, 2, 5]
            for i in order:
                if mode == 'events': a.ravel()[ids[i]] += 1
                else: a.reshape(32, 4)[:, ids[i] % 4] = full.reshape(32, 4)[:, ids[i] % 4]
                p = encode_snapshot(a, m)
                candidates.append((p, a.copy()))
                boundaries += [len(p)-1, len(p), len(p)+1]
            for budget in boundaries:
                expected = b''; counts = np.zeros_like(full)
                for p, a in candidates:
                    if len(p) > budget: break
                    expected, counts = p, a
                p, a, report = select_snapshot(ids, m, budget, mode)
                self.assertEqual(p, expected)
                np.testing.assert_array_equal(a, counts)
                self.assertLessEqual(report['actual_bytes'], budget)

    def test_empty_and_invalid_stream(self):
        m = metadata()
        for mode in ('events', 'rows'):
            p, a, r = select_snapshot(np.array([], dtype=np.uint32), m, 0, mode)
            self.assertEqual(p, b''); self.assertEqual(a.sum(), 0)
            self.assertEqual(r['attempted_prefixes'], 0)
        for ids in (np.array([128]), np.array([-1]), np.zeros(9, dtype=np.uint32)):
            with self.assertRaises(ValueError): select_snapshot(ids, m, 2048, 'events')


if __name__ == '__main__': unittest.main()

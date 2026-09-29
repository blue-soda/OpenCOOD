import unittest
from opencood.data_utils.int_sequence_order import segment_order


class TestSegmentOrder(unittest.TestCase):
    def test_no_truncation_or_unlabeled_warmup_loss(self):
        rows = [dict(frame=str(i), segment=str(i//9), timestamp_us=i*100000,
                     supervised=i%9==8, reset_before=False) for i in range(27)]
        result = segment_order(rows, 42)
        self.assertEqual(sorted(r['frame'] for r in result), sorted(r['frame'] for r in rows))
        self.assertEqual(sum(r['reset_before'] for r in result), 3)
        for start in range(0, 27, 9):
            group = result[start:start+9]
            self.assertEqual(len(set(r['segment'] for r in group)), 1)
            self.assertTrue(group[0]['reset_before'])
            self.assertTrue(group[-1]['supervised'])
            self.assertEqual([r['timestamp_us'] for r in group], sorted(r['timestamp_us'] for r in group))
        self.assertFalse(any(r['reset_before'] for r in rows))
        self.assertEqual(result, segment_order(rows, 42))

    def test_reject_duplicate_and_noncausal(self):
        row = dict(frame='a', segment='s', timestamp_us=1)
        with self.assertRaises(ValueError): segment_order([row, row], 1)
        with self.assertRaises(ValueError): segment_order([row, dict(row, frame='b', timestamp_us=0)], 1)


if __name__ == '__main__':
    unittest.main()

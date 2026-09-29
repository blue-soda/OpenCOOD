import unittest
from opencood.tools.audit_int_stream import filter_stream


class TestDecodeBoundaries(unittest.TestCase):
    def test_bad_scan_splits_memory_without_changing_other_segments(self):
        rows = [dict(frame=str(i), segment='a' if i < 3 else 'b') for i in range(4)]
        audits = [dict(frame=str(i), reason='empty' if i == 1 else None) for i in range(4)]
        result = filter_stream(rows, audits)
        self.assertEqual([x['frame'] for x in result], ['0', '2', '3'])
        self.assertNotEqual(result[0]['segment'], result[1]['segment'])
        self.assertEqual(result[2], rows[3])
        self.assertEqual(rows[2]['segment'], 'a')

    def test_valid_stream_unchanged_and_incomplete_audit_rejected(self):
        rows = [dict(frame='0', segment='a')]
        self.assertEqual(filter_stream(rows, [dict(frame='0', reason=None)]), rows)
        with self.assertRaises(ValueError): filter_stream(rows, [])


if __name__ == '__main__':
    unittest.main()

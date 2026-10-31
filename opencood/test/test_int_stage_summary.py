import unittest
from opencood.tools.int_stage_summary import stage_means


class StageSummaryTests(unittest.TestCase):
    def test_one_two_and_three_stage_wrappers(self):
        for count in (1, 2, 3):
            records = [dict(cache_stages=[dict(write=value+i) for i in range(count)])
                       for value in (0., 1.)]
            self.assertEqual(stage_means(records), [dict(write=.5+i) for i in range(count)])

    def test_changed_schema_is_rejected(self):
        for changed in ([], [dict(other=1.)], [dict(write=1.), dict(write=1.)]):
            with self.assertRaises(ValueError):
                stage_means([dict(cache_stages=[dict(write=0.)]), dict(cache_stages=changed)])

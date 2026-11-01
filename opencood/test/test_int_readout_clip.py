import unittest
import torch
from opencood.tools.diagnose_int_readout_clip import tail_stats


class TailStatisticsTests(unittest.TestCase):
    def test_known_tail_and_spatial_concentration(self):
        x=torch.tensor([[[[0.,1.],[2.,4.]],[[1.,1.],[1.,1.]]]])
        result=tail_stats(x,[1.,2.]);s=result['summary']
        self.assertEqual(s['clipped_count'],2)
        self.assertEqual(s['activation_sum'],11.)
        self.assertEqual(s['removed_sum'],4.)
        self.assertEqual(s['removed_square_sum'],10.)
        self.assertEqual(s['activation_square_sum'],25.)
        self.assertEqual(s['clipped_spatial_cells'],2)
        self.assertAlmostEqual(s['top1pct_cells_removed_square_fraction'],.9)
        self.assertEqual(s['maximum_scale_ratio'],4.)
        self.assertEqual(result['channels']['maximum'],[4.,1.])

    def test_no_tail_has_finite_zero_concentration(self):
        s=tail_stats(torch.zeros(1,2,2,2),[1.,1.])['summary']
        self.assertEqual(s['removed_square_sum'],0.)
        self.assertEqual(s['top1pct_cells_removed_square_fraction'],0.)


if __name__=='__main__':unittest.main()

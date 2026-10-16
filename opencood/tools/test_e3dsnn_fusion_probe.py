import unittest
import torch
from opencood.utils.e3dsnn_fusion_probe import probe_maps, summarize_probe, detector_digest


class ProbeTest(unittest.TestCase):
    def test_count_dead_zone_and_clipping(self):
        ego = torch.tensor([1., 4., 0., 2.]).reshape(1, 4, 1, 1)
        residual = torch.tensor([.2, 1., -.3, .6]).reshape_as(ego)
        fused = (ego + residual).clamp(0, 4).add(.5).floor()
        maps = probe_maps(ego, residual, fused, torch.ones(1, 1, 1, 1), 'count4')
        self.assertEqual(float(maps['changed_channel_fraction']), .25)
        self.assertEqual(float(maps['nonzero_residual_unchanged_fraction']), .75)
        self.assertEqual(float(maps['above_cap_fraction']), .25)
        self.assertEqual(float(maps['below_zero_fraction']), .25)

    def test_unsupported_roi_is_missing_not_zero(self):
        x = torch.ones(1, 2, 2, 2)
        maps = probe_maps(x, x * 0, x, torch.zeros(1, 1, 2, 2), 'relu')
        out = summarize_probe(maps, torch.zeros(1, 8, 3), [0, 0, 0], [1, 1, 1])
        self.assertEqual(out['gt_regions'][0]['supported_cells'], 0)
        self.assertIsNone(out['frame']['changed_channel_fraction'])

    def test_hash_covers_bn_buffers_and_excludes_fuser(self):
        model = torch.nn.Module()
        model.detector = torch.nn.BatchNorm1d(2)
        model.fusion_net = torch.nn.Linear(2, 2)
        before = detector_digest(model)
        with torch.no_grad(): model.fusion_net.weight.add_(1)
        self.assertEqual(before, detector_digest(model))
        model.detector.running_mean.add_(1)
        self.assertNotEqual(before, detector_digest(model))


if __name__ == '__main__':
    unittest.main()

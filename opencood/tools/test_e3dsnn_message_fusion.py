import unittest
import torch
from opencood.models.fuse_modules.e3dsnn_geometry import warp_volume
from opencood.models.fuse_modules.e3dsnn_message_support import message_support
from opencood.models.fuse_modules.e3dsnn_readout import residual_readout


class MessageSupportTest(unittest.TestCase):
    def test_identity_empty_and_geometry_contract(self):
        road = torch.zeros(2, 3, 3, 5, 5)
        road[0, 1, 1, 2, 2] = 4
        transform = torch.eye(4).repeat(2, 1, 1)
        _, coverage = warp_volume(road, transform, [0]*3, [1]*3)
        self.assertIs(message_support(road, transform, coverage, [0]*3, [1]*3, 'geometry'), coverage)
        mask = message_support(road, transform, coverage, [0]*3, [1]*3, 'received')
        self.assertTrue(torch.equal(mask, road.ne(0).any(1, keepdim=True).float()))
        support = mask.flatten(1, 2).amax(1, keepdim=True)
        ego = torch.rand(2, 9, 5, 5) * 4
        residual = support * torch.ones_like(ego)
        result = residual_readout(ego, residual, support, torch.nn.ReLU())
        self.assertTrue(torch.equal(result[1], ego[1]))
        self.assertTrue(torch.equal(result[support.expand_as(ego) == 0], ego[support.expand_as(ego) == 0]))
        self.assertFalse(torch.equal(result[0, :, 2, 2], ego[0, :, 2, 2]))

    def test_fractional_translation_and_domain(self):
        road = torch.zeros(1, 2, 3, 5, 5, dtype=torch.double)
        road[0, 0, 1, 2, 2] = 3
        transform = torch.eye(4, dtype=torch.double)[None]
        transform[0, 0, 3] = .5
        _, coverage = warp_volume(road, transform, [0]*3, [1]*3)
        mask = message_support(road, transform, coverage, [0]*3, [1]*3, 'received')
        self.assertAlmostEqual(float(mask[0, 0, 1, 2, 2]), .5)
        self.assertAlmostEqual(float(mask[0, 0, 1, 2, 3]), .5)
        self.assertAlmostEqual(float(mask.sum()), 1.)
        self.assertTrue(torch.all(mask <= coverage))
        transform[0, 0, 3] = 100
        _, coverage = warp_volume(road, transform, [0]*3, [1]*3)
        mask = message_support(road, transform, coverage, [0]*3, [1]*3, 'received')
        self.assertEqual(float(mask.sum()), 0.)

    def test_decoded_count_support_independent_of_amplitude(self):
        road = torch.zeros(1, 2, 3, 5, 5)
        road[0, :, 1, 2, 2] = 1
        transform = torch.eye(4)[None]
        _, coverage = warp_volume(road, transform, [0]*3, [1]*3)
        a = message_support(road, transform, coverage, [0]*3, [1]*3, 'received')
        b = message_support(road * 4, transform, coverage, [0]*3, [1]*3, 'received')
        self.assertTrue(torch.equal(a, b))

    def test_matched_pipeline_and_message_drift_guard(self):
        from pathlib import Path
        from opencood.tools.launch_e3dsnn_message_fusion import GROUPS, make_jobs, check_bytes
        root = Path('/audit/message')
        self.assertEqual(len(GROUPS), 4)
        for g in GROUPS:
            name, jobs = make_jobs(root, g, 'adapt10')
            self.assertEqual(name, 'message_'+g+'_adapt10')
            self.assertEqual(len(jobs), 6)
            cmd = jobs[0]['command']
            self.assertEqual(cmd[cmd.index('--epochs')+1], '10')
            self.assertEqual(cmd[cmd.index('--config')+1], str(root/(g+'.yaml')))
            self.assertEqual(cmd[cmd.index('--checkpoint')+1], str(root/(g+'_initial.pth')))
            self.assertEqual(jobs[3]['command'][jobs[3]['command'].index('--config')+1], str(root/(g+'_ego.yaml')))
            self.assertEqual(len(make_jobs(root,g,'zero')[1]), 2)
        a = dict(communication_frames=[dict(frame_id='1',bytes=100)])
        self.assertEqual(check_bytes(a,a)['mean_absolute_frame_bytes'], 0)
        with self.assertRaises(AssertionError):
            check_bytes(a,dict(communication_frames=[dict(frame_id='1',bytes=110)]))


if __name__ == '__main__': unittest.main()

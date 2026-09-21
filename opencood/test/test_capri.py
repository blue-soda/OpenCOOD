import unittest
import torch
from opencood.models.capri.geometry import fit_se2, move_keypoints, register_background
from opencood.models.capri.model import Capri, capri_loss


def message(x=0., count=1):
    boxes = torch.tensor([[x, 0., 0., 2., 2., 4., 0.]]).repeat(count, 1)
    return {'boxes': boxes, 'scores': torch.full((count,), .8),
            'points': boxes[:, None, :3].repeat(1, 4, 1)+torch.randn(count, 4, 3)*.1,
            'features': torch.randn(count, 4, 8), 'valid': torch.ones(count, 4, dtype=torch.bool),
            'background': {'xyz': torch.zeros(0, 3), 'descriptor': torch.zeros(0, 8), 'quality': torch.zeros(0)}}


class CapriTest(unittest.TestCase):
    def model(self):
        return Capri(8, {'hidden_dim': 16, 'max_acceleration': 6., 'match_radius': 5.,
                         'max_missing_age_s': 2., 'fusion_radius': 6.})

    def test_se2_known_transform(self):
        torch.manual_seed(1)
        x = torch.randn(30, 3)
        angle = torch.tensor(.1)
        r = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
        y = x.clone()
        y[:, :2] = x[:, :2] @ r.T + torch.tensor([.4, -.3])
        transform, _ = fit_se2(x, y, torch.ones(30))
        torch.testing.assert_allclose(transform[:2, :2], r)
        torch.testing.assert_allclose(transform[:2, 3], torch.tensor([.4, -.3]))

    def test_empty_registration_and_zero_dt(self):
        m = message()
        transform, stats = register_background(m['background'], m['background'])
        self.assertEqual(stats['accepted'], 0.)
        torch.testing.assert_allclose(transform, torch.eye(4))
        model = self.model()
        state = model.initialize(m, -.3, 0.)
        result = model.propagate(state, 0.)
        torch.testing.assert_allclose(state['boxes'], result['boxes'])
        torch.testing.assert_allclose(state['points'], result['points'])

    def test_rigid_keypoint_motion(self):
        old = message()['boxes']
        new = old.clone()
        new[:, :2] += 2
        points = old[:, None, :3].repeat(1, 4, 1)
        torch.testing.assert_allclose(move_keypoints(points, old, new), new[:, None, :3].repeat(1, 4, 1))

    def test_birth_missing_and_gradient(self):
        model = self.model()
        state, loss, counts = model.sequence([message(.2), message(0.)], [-.1, -.2], [0., 0.])
        self.assertEqual(counts['matched'], 1)
        self.assertAlmostEqual(float(state['boxes'][0, 0]), .4, places=5)
        messages = [message(), message(), message(.2), message()]
        output = model(messages, torch.eye(4).repeat(2, 2, 1, 1),
                       torch.tensor([[0., 0.], [-.1, -.2]]), [message(), message()],
                       torch.eye(4).repeat(2, 1, 1), torch.tensor([False, False]))
        objective, _ = capri_loss(output, message(.5)['boxes'])
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.motion.parameters() if p.grad is not None), 0)
        state, _, counts = model.sequence([message(count=0), message()], [-.1, -.2], [0., 0.])
        self.assertEqual(counts['retained_missing'], 1)
        self.assertEqual(len(state['boxes']), 1)


if __name__ == '__main__':
    unittest.main()

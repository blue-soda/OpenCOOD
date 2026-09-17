import unittest
import torch
from torch import nn

from opencood.models.sub_modules.ectra_recurrent_alignment import EctraRecurrentAlignment
from opencood.models.sub_modules.ectra_roi_flow_refiner import EctraRoiFlowRefiner
from opencood.tools.ectra_ablation_utils import ABLATIONS, install_ablation


def make_model():
    model = nn.Module()
    model.ectra = EctraRecurrentAlignment({'feature_dim': 2, 'hidden_dim': 4,
                                         'state_update_mode': 'residual_observation'})
    model.ectra_roi = EctraRoiFlowRefiner({'feature_dim': 2, 'hidden_dim': 4})
    return model.eval()


class AblationTest(unittest.TestCase):
    def test_all_recurrent_interventions_have_finite_outputs(self):
        for mode in ABLATIONS:
            with self.subTest(mode=mode):
                model = make_model()
                install_ablation(model, mode)
                x = torch.rand(4, 2, 4, 5)
                output, _ = model.ectra(
                    x, torch.tensor([2]), torch.tensor([0., 0., -3., -6.]))
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(torch.isfinite(output).all())
                if mode == 'recurrent_bypass':
                    torch.testing.assert_allclose(output, x)

    def test_motion_identity_and_checkpoint_unchanged(self):
        model = make_model()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        install_ablation(model, 'motion_identity')
        x = torch.rand(1, 2, 4, 5)
        pred, flow, gamma = model.ectra._propagate(x, x, torch.tensor([3.]))
        torch.testing.assert_allclose(pred, x)
        self.assertEqual(flow.abs().sum().item(), 0)
        self.assertEqual(gamma.min().item(), 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_allclose(value, saved[key])

    def test_roi_bypasses_are_separate(self):
        features = torch.randn(2, 2, 4, 5)
        for mode in ('roi_refiner_bypass', 'roi_flow_bypass', 'roi_trust_bypass'):
            model = make_model()
            grid = model.ectra_roi._identity_grid(2, 4, 5, features.device, features.dtype)
            mask = torch.ones_like(features)
            install_ablation(model, mode)
            actual_grid, actual_mask, _ = model.ectra_roi(
                grid, mask, features, torch.tensor([2]), torch.tensor([0., -3.]))
            if mode != 'roi_trust_bypass':
                torch.testing.assert_allclose(actual_grid, grid)
            if mode != 'roi_flow_bypass':
                torch.testing.assert_allclose(actual_mask, mask)
            if mode == 'roi_flow_bypass':
                self.assertLess(actual_mask[1].max().item(), 1)

    def test_training_interventions_rejected(self):
        with self.assertRaises(ValueError):
            install_ablation(make_model().train(), 'calib_identity')


if __name__ == '__main__':
    unittest.main()

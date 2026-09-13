"""Small CPU regressions for ECTRA delay encoding and auxiliary losses."""
import importlib.util
import os
from pathlib import Path
import unittest

import torch


def load_module(name, relative):
    override = os.environ.get('ECTRA_AUDIT_MODULE_DIR')
    path = Path(override) / Path(relative).name if override else Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EctraRegressionTest(unittest.TestCase):
    def test_decoder_bn_separates_statistics_with_shared_affine(self):
        cls = load_module('domain', 'models/sub_modules/decoder_domain_norm.py').DecoderDomainBatchNorm2d
        bn = cls(torch.nn.BatchNorm2d(2, momentum=1.)).train()
        bn(torch.full((2, 2, 3, 3), 2.))
        bn.fusion_domain = bn.fusion_training = True
        bn(torch.full((2, 2, 3, 3), 8.))
        torch.testing.assert_allclose(bn.running_mean, torch.full((2,), 2.))
        torch.testing.assert_allclose(bn.fusion_running_mean, torch.full((2,), 8.))
        self.assertEqual(sum(p.numel() for p in bn.parameters()), 4)

    def test_roi_identity_grid_preserves_pixels_and_flow_has_gradient(self):
        cls = load_module('roi', 'models/sub_modules/ectra_roi_flow_refiner.py').EctraRoiFlowRefiner
        model = cls({'feature_dim': 2, 'hidden_dim': 4}).eval()
        features = torch.randn(2, 2, 4, 5)
        grid = model._identity_grid(2, 4, 5, features.device, features.dtype)
        torch.testing.assert_allclose(torch.nn.functional.grid_sample(features, grid, align_corners=False), features)
        refined, mask, aux = model(grid, torch.ones_like(features), features, torch.tensor([2]), torch.tensor([0., -3.]))
        sampled = torch.nn.functional.grid_sample(features, refined, align_corners=False) * mask
        sampled.square().mean().backward()
        self.assertGreater(model.flow_head.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.trust_head.weight.grad.abs().sum().item(), 0.)

    def test_signed_offsets_encode_elapsed_time(self):
        cls = load_module('roi', 'models/sub_modules/ectra_roi_flow_refiner.py').EctraRoiFlowRefiner
        model = cls({'feature_dim': 2, 'hidden_dim': 4}).eval()
        inputs = []
        hook = model.encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
        features = torch.randn(2, 2, 4, 5)
        grid = model._identity_grid(2, 4, 5, features.device, features.dtype)
        for delay in (-3., 3., 0.):
            model(grid, torch.ones_like(features), features, torch.tensor([2]), torch.tensor([0., delay]))
        hook.remove()
        torch.testing.assert_allclose(inputs[0], inputs[1])
        self.assertGreater(inputs[0][1, -2:].abs().sum().item(), 0.)
        self.assertEqual(inputs[2][1, -2:].abs().sum().item(), 0.)

    def test_aux_total_gradient_and_missing_reset(self):
        cls = load_module('tc', 'loss/point_pillar_tc_loss.py').PointPillarTcLoss
        criterion = cls({'cls_weight': 1., 'reg': 2., 'aux_weights': {'ectra_motion_loss': .5}})
        output = {'psm': torch.zeros(1, 2, 1, 1), 'rm': torch.zeros(1, 14, 1, 1)}
        target = {'targets': torch.zeros(1, 1, 1, 14), 'pos_equal_one': torch.ones(1, 1, 1, 2)}
        base = criterion(output, target).detach()
        aux = torch.tensor(2., requires_grad=True)
        loss = criterion(dict(output, ectra_motion_loss=aux), target)
        torch.testing.assert_allclose(loss, base + 1.)
        torch.testing.assert_allclose(criterion.loss_dict['total_loss'], loss)
        loss.backward()
        self.assertAlmostEqual(aux.grad.item(), .5)
        criterion(output, target)
        self.assertNotIn('ectra_motion_loss', criterion.loss_dict)


if __name__ == '__main__':
    unittest.main()

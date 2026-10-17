import unittest
import torch
from opencood.utils.e3dsnn_transplant import compose_state, encoder_digest


class TransplantTest(unittest.TestCase):
    def test_partition_copies_bn_and_heads_without_aliasing(self):
        keys = ['backbone_3d.weight', 'backbone_3d.running_mean', 'fusion_scale',
                'fusion_net.weight', 'backbone_2d.running_var', 'cls_head.weight', 'reg_head.weight', 'dir_head.weight']
        a = {k: torch.tensor([1.]) for k in keys}
        b = {k: torch.tensor([2.]) for k in keys}
        mixed = compose_state(a, b)
        for k in keys:
            self.assertEqual(float(mixed[k]), 1. if k.startswith('backbone_3d.') else 2.)
        mixed['reg_head.weight'].add_(1)
        self.assertEqual(float(b['reg_head.weight']), 2.)

    def test_reject_unknown_or_incompatible_state(self):
        for a, b in [({'other': torch.ones(1)}, {'other': torch.ones(1)}),
                     ({'reg_head.weight': torch.ones(1)}, {'reg_head.weight': torch.ones(2)}),
                     ({'reg_head.weight': torch.ones(1)}, {})]:
            with self.assertRaises(ValueError): compose_state(a, b)

    def test_encoder_audit_covers_buffers_and_ignores_receiver(self):
        model = torch.nn.Module()
        model.backbone_3d = torch.nn.BatchNorm1d(2)
        model.reg_head = torch.nn.Linear(2, 2)
        original = encoder_digest(model)
        with torch.no_grad(): model.reg_head.weight.add_(1)
        self.assertEqual(original, encoder_digest(model))
        model.backbone_3d.running_mean.add_(1)
        self.assertNotEqual(original, encoder_digest(model))


if __name__ == '__main__':
    unittest.main()

"""CPU admission for source modulation, IF ordering and actual byte roundtrip.

These synthetic tests do not replace full-model/data admission or AP validation.
"""
import copy
from dataclasses import asdict
import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_spatial_drive import SnapshotSpatialDrive
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import select_training_snapshot
from opencood.utils import task_snapshot_packet as wire


class SpatialDriveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.features = torch.rand(1, 128, 2, 3, 4)
        self.drive = torch.rand(1, 32, 2, 3, 4) * 5

    def test_identity_capacity_rng_and_zero_support(self):
        rng = torch.get_rng_state().clone()
        arms = [SnapshotSpatialDrive(c) for c in ('spatial', 'global')]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in arms[0].parameters()), 55329)
        self.assertTrue(all(torch.equal(v, arms[1].state_dict()[k])
                            for k, v in arms[0].state_dict().items()))
        for module in arms:
            self.assertTrue(torch.equal(module(self.features, self.drive), self.drive))
            with torch.no_grad(): module.net[-1].bias.fill_(2)
            self.assertTrue(torch.equal(module(self.features, self.drive * 0), self.drive * 0))

    def test_spatial_changes_if_order_and_global_broadcasts(self):
        a = SnapshotSpatialDrive('spatial')
        with torch.no_grad():
            for p in a.parameters(): p.zero_()
            a.net[0].weight[0, 0, 1, 1, 1] = 1
            a.net[-1].weight[0, 0, 0, 0, 0] = 1
            a.net[-1].bias.fill_(-1)
        b = SnapshotSpatialDrive('global'); b.load_state_dict(a.state_dict(), strict=True)
        features = torch.zeros(1, 128, 1, 1, 2)
        features[0, 0, 0, 0] = torch.tensor([.1, 3.])
        drive = torch.ones(1, 32, 1, 1, 2) * 1.5
        spatial, glob = a(features, drive), b(features, drive)
        self.assertTrue(torch.equal(glob[..., 0], glob[..., 1]))
        self.assertEqual(int(integrate_and_fire(glob).addresses[0]), 0)
        self.assertEqual(int(integrate_and_fire(spatial).addresses[0]), 1)
        # Constant logits must collapse to the same global intervention.
        with torch.no_grad(): a.net[-1].weight.zero_(); b.net[-1].weight.zero_()
        self.assertTrue(torch.equal(a(features, drive), b(features, drive)))

    def test_two_step_surrogate_gradient(self):
        for condition in ('spatial', 'global'):
            module = SnapshotSpatialDrive(condition)
            opt = torch.optim.SGD(module.parameters(), lr=.01)
            for step in range(2):
                opt.zero_grad()
                stream = integrate_and_fire(module(self.features, self.drive))
                stream.prefix(min(100, len(stream.addresses))).square().mean().backward()
                self.assertGreater(float(module.net[-1].weight.grad.abs().sum()), 0)
                hidden = float(module.net[0].weight.grad.abs().sum())
                if step == 0: self.assertEqual(hidden, 0)
                else: self.assertGreater(hidden, 0)
                opt.step()

    def test_basis_identity_and_real_budget_roundtrip(self):
        module = SnapshotSpatialDrive('spatial')
        initial = module.basis_digest('0' * 64)
        same = copy.deepcopy(module)
        self.assertEqual(initial, same.basis_digest('0' * 64))
        same.condition = 'global'
        self.assertNotEqual(initial, same.basis_digest('0' * 64))
        self.assertNotEqual(initial, module.basis_digest('1' * 64))
        with torch.no_grad(): module.net[-1].bias.add_(.1)
        self.assertNotEqual(initial, module.basis_digest('0' * 64))
        features = self.features[:, :, :1, :1, :4]
        drive = self.drive[:, :, :1, :1, :4]
        stream = integrate_and_fire(module(features, drive))
        full = stream.prefix(len(stream.addresses)); scales = full.new_ones(1, 32, 1, 1, 1)
        contract = MessageContract('ego', 'road', 'infrastructure', 'a' * 64,
            'task32-' + module.basis_digest('0' * 64), (1, 1, 4), (0., 0., 0.),
            (1., 1., 1.), (123, 124))
        meta = dict(contract=asdict(contract), source_to_ego=np.eye(4).tolist(),
                    family='if8', schema='task-count32-v1')
        full_packet = wire.encode(full.detach().numpy(), np.ones(32, dtype=np.float32), meta)
        for budget in (0, len(full_packet) - 1, len(full_packet), 2304):
            chosen = select_training_snapshot(full, scales, meta, budget, stream=stream)
            restored, _ = wire.decode(chosen.packet, meta)
            np.testing.assert_array_equal(restored, chosen.counts.detach().numpy())
            self.assertLessEqual(len(chosen.packet), budget)
            if budget == 0: self.assertEqual(float(chosen.counts.sum()), 0)
        wrong = copy.deepcopy(meta); wrong['contract']['channel_basis'] = 'task32-' + initial
        with self.assertRaises(ValueError): wire.decode(full_packet, wrong)

    def test_invalid_condition_and_shapes(self):
        with self.assertRaises(ValueError): SnapshotSpatialDrive('unknown')
        m = SnapshotSpatialDrive('spatial')
        with self.assertRaises(ValueError): m(self.features[:, :127], self.drive)
        with self.assertRaises(ValueError): m(self.features, self.drive[:, :31])
        with self.assertRaises(ValueError): m(self.features, self.drive[..., :2])


if __name__ == '__main__': unittest.main()

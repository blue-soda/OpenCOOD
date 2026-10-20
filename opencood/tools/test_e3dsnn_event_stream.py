"""Event timing, transport, geometry and surrogate-gradient reference tests."""
import dataclasses
import unittest
import numpy as np
import torch
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
from opencood.models.fuse_modules.e3dsnn_joint_membrane import JointMembraneReceiver
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
from opencood.utils import event_stream_packet as wire


def metadata():
    c = MessageContract('car', 'road', 'infra', 'a'*64, 'event-content32-'+'b'*64,
                        (1, 1, 4), (0., 0., 0.), (1., 1., 1.), (10, 12))
    return dict(contract=dataclasses.asdict(c), source_to_ego=np.eye(4).tolist(),
                channels=32, max_spikes=8, schema='if-unit-events-v1')


class StreamTest(unittest.TestCase):
    def test_crossings_repeated_channel_and_conservation(self):
        x = torch.tensor([2.5, 1.25, 0., .7]).reshape(1, 1, 1, 1, 4)
        s = integrate_and_fire(x)
        self.assertEqual(s.addresses.tolist(), [0, 0, 1])
        np.testing.assert_allclose(s.times, [.4, .8, .8])
        self.assertTrue(torch.equal(s.prefix(3), x.floor()))
        residual = x - s.prefix(3)
        self.assertTrue(torch.all((residual >= 0) & (residual < 1)))
        self.assertTrue(torch.equal(s.prefix(1), torch.tensor([1., 0, 0, 0]).reshape_as(x)))

    def test_silent_and_invalid(self):
        x = torch.zeros(1, 32, 1, 1, 4)
        self.assertEqual(len(integrate_and_fire(x).addresses), 0)
        self.assertEqual(wire.pack_prefix(np.array([], dtype='<u4'), metadata(), 1000), ([], 0, 0))
        for bad in (-x-1, x+float('nan')):
            with self.assertRaises(ValueError):
                integrate_and_fire(bad)
        with self.assertRaises(ValueError):
            integrate_and_fire(x, 0)

    def test_budget_includes_header_and_variable_packet_count(self):
        ids = np.arange(128, dtype='<u4')
        full = wire.encode(ids[:1], metadata(), 0)
        for budget in (0, len(full)-1, len(full), 1500, 5000):
            packets, n, size = wire.pack_prefix(ids, metadata(), budget, packet_events=17)
            self.assertEqual(size, sum(map(len, packets)))
            self.assertLessEqual(size, budget)
            acc = wire.StreamAccumulator(metadata())
            for packet in packets:
                acc.receive(packet)
            self.assertEqual(int(acc.finish().sum()), n)
        packets, n, _ = wire.pack_prefix(ids, metadata(), 100000, packet_events=17)
        self.assertEqual(len(packets), 8)
        self.assertEqual(n, 128)

    def test_repack_same_events_exact_counts_and_geometry(self):
        ids = np.array([1, 1, 5, 5, 5, 10], dtype='<u4')
        results = []
        for capacity in (1, 2, 6, 256):
            packets, n, _ = wire.pack_prefix(ids, metadata(), 100000, capacity)
            acc = wire.StreamAccumulator(metadata())
            for p in packets:
                acc.receive(p)
            results.append(acc.finish())
            self.assertEqual(n, len(ids))
        self.assertTrue(all(np.array_equal(results[0], r) for r in results))
        receiver = JointMembraneReceiver(1, 32).double()
        with torch.no_grad():
            receiver.current_scale.fill_(.4)
        pose = torch.eye(4, dtype=torch.double)[None]; pose[0, 0, 3] = .5
        c = wire.validate_metadata(metadata())
        u = torch.zeros(1, 1, 1, 1, 4, dtype=torch.double)
        out = [stream_readout(u, torch.tensor(r, dtype=torch.double), pose, c,
                              receiver, lambda x: x.sum(1, keepdim=True)) for r in results]
        self.assertTrue(all(torch.equal(out[0], r) for r in out))
        self.assertEqual(float(out[0][..., 0]), 0.)
        self.assertGreater(float(out[0][..., 2]), 0.)

    def test_reject_replay_pose_basis_frame_and_truncation(self):
        m = metadata(); p = wire.encode(np.array([1, 1], dtype='<u4'), m, 0)
        acc = wire.StreamAccumulator(m); acc.receive(p)
        with self.assertRaises(ValueError):
            acc.receive(p)
        for key in ('receiver_frame', 'channel_basis'):
            other = metadata(); other['contract'][key] += 'wrong'
            with self.assertRaises(ValueError):
                acc.receive(wire.encode(np.array([1], dtype='<u4'), other, 2))
        other = metadata(); other['source_to_ego'][0][3] = 1.
        with self.assertRaises(ValueError):
            acc.receive(wire.encode(np.array([1], dtype='<u4'), other, 2))
        with self.assertRaises(ValueError):
            wire.decode(p[:-1])
        with self.assertRaises(ValueError):
            acc.receive(wire.encode(np.ones(7, dtype='<u4'), m, 2))
        self.assertEqual(acc.offset, 2)
        acc.finish()
        with self.assertRaises(ValueError):
            acc.receive(wire.encode(np.array([2], dtype='<u4'), m, 2))

    def test_fixed_frontier_surrogate_and_ceiling(self):
        x = torch.tensor([2.5, 1.25, .7, 10.], requires_grad=True)
        s = integrate_and_fire(x.reshape(1, 1, 1, 1, 4), max_spikes=4)
        y = s.prefix(3)
        self.assertTrue(torch.equal(y, y.floor()))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:3].sum()), 0)
        self.assertEqual(float(x.grad[3]), 0)


if __name__ == '__main__':
    unittest.main()

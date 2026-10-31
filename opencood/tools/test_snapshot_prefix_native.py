"""Exact native/reference prefix, packet, and STE gradient comparisons."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from opencood.tools.test_e3dsnn_task_snapshot import meta
from opencood.utils import task_snapshot_packet as wire
from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire
from opencood.models.fuse_modules.e3dsnn_snapshot_selection import select_training_snapshot


class NativePrefixTest(unittest.TestCase):
    def setUp(self):
        self.library = os.environ['OPENCOOD_SNAPSHOT_NATIVE_LIB']

    def compare(self, counts, ids, budget):
        scales = np.ones(32, dtype=np.float32)
        with patch.dict(os.environ, OPENCOOD_SNAPSHOT_NATIVE_LIB=''):
            expected = wire.select(counts, scales, meta('if8'), budget, addresses=ids)
        actual = wire.select(counts, scales, meta('if8'), budget, addresses=ids)
        self.assertEqual(actual[0], expected[0]); self.assertEqual(actual[2], expected[2])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_every_boundary_and_multiple_threads(self):
        rng = np.random.RandomState(53)
        for threads in ('1', '2', '8'):
            with patch.dict(os.environ, OPENCOOD_SNAPSHOT_THREADS=threads, OPENCOOD_SNAPSHOT_CHUNK='32'):
                for _ in range(3):
                    ids = rng.permutation(np.tile(np.arange(128), 2))[:160]
                    counts = np.bincount(ids, minlength=128).reshape(1,32,1,1,4)
                    running = np.zeros_like(counts); caps = {0, 8192}
                    for addr in ids:
                        running.ravel()[addr] += 1
                        cap = len(wire.encode(running, np.ones(32,dtype=np.float32), meta('if8')))
                        caps.update((cap-1, cap, cap+1))
                    for cap in sorted(caps): self.compare(counts, ids, cap)
                self.compare(np.zeros((1,32,1,1,4)), np.array([],dtype=np.int64), 0)

    def test_gradient_exact(self):
        def run(native, budget):
            with patch.dict(os.environ, OPENCOOD_SNAPSHOT_NATIVE_LIB=self.library if native else ''):
                torch.manual_seed(7)
                drive = (torch.rand(1,32,1,1,4)*4).requires_grad_()
                stream = integrate_and_fire(drive,8); full = stream.prefix(len(stream.addresses))
                result = select_training_snapshot(full,torch.ones(1,32,1,1,1),meta('if8'),budget,stream=stream)
                result.counts.square().sum().backward()
                return result, drive.grad
        for cap in (0, 1000, 8192):
            old, og = run(False, cap); new, ng = run(True, cap)
            self.assertEqual(old.packet,new.packet); self.assertEqual(old.record,new.record)
            self.assertTrue(torch.equal(old.counts,new.counts)); self.assertTrue(torch.equal(og,ng))

    def test_sequential_validation_rng_draw(self):
        torch.manual_seed(91); before = torch.get_rng_state()
        loader = DataLoader(TensorDataset(torch.arange(4)),batch_size=1,shuffle=False,num_workers=0)
        list(loader); after = torch.get_rng_state()
        torch.set_rng_state(before); torch.empty((),dtype=torch.int64).random_()
        self.assertTrue(torch.equal(after,torch.get_rng_state()))


if __name__ == '__main__': unittest.main()

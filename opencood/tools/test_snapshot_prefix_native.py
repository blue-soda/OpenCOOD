"""Exact native/reference prefix, packet, and STE gradient comparisons."""
import os
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
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
        for workers in (0,2):
            torch.manual_seed(91); before = torch.get_rng_state()
            loader = DataLoader(TensorDataset(torch.arange(4)),batch_size=1,shuffle=False,num_workers=workers,
                **({'multiprocessing_context':'spawn'} if workers else {}))
            list(loader); after = torch.get_rng_state()
            torch.set_rng_state(before); torch.empty((),dtype=torch.int64).random_()
            self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_deferred_selection_keeps_actual_initial_weight(self):
        from opencood.utils.deferred_initial_validation import collect
        from opencood.tools.train_e3dsnn_single import save_best_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=root/'initial.pth';torch.save({'weight':torch.tensor([3.])},str(checkpoint))
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest=dict(checkpoint_sha256=digest,config_sha256='config',data_manifest_sha256='data',val_samples=2)
            (root/'manifest_1.json').write_text(json.dumps(manifest))
            (root/'status.json').write_text(json.dumps(dict(state='complete')))
            (root/'evaluation.json').write_text(json.dumps(dict(full_split=True,samples=2,ap={'0.5':.7,'0.7':.6})))
            handle=(SimpleNamespace(wait=lambda timeout:0),root,dict(started_unix=time.time(),checkpoint_sha256=digest))
            initial,metrics=collect(handle,str(checkpoint),manifest)
            bests={'0.5':-1.,'0.7':-1.};best=save_best_checkpoints(initial,metrics,root,bests,'0.7',-1.)
            trained=SimpleNamespace(state_dict=lambda:{'weight':torch.tensor([9.])})
            best=save_best_checkpoints(trained,dict(ap={'0.5':.71,'0.7':.59},epoch=1),root,bests,'0.7',best)
            self.assertEqual(torch.load(str(root/'best.pth'))['weight'].item(),3.)
            self.assertEqual(torch.load(str(root/'best_ap50.pth'))['weight'].item(),9.)
            with self.assertRaises(AssertionError):collect(handle,str(checkpoint),dict(manifest,config_sha256='wrong'))


if __name__ == '__main__': unittest.main()

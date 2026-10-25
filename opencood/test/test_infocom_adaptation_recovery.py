"""Check recovery determinism and failure-before-replace, using a tiny model."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]/'tools/strong_ann_baselines'
sys.path.insert(0,str(SOURCE))
import train_infocom_lowrate as training


class RecoveryTests(unittest.TestCase):
    def test_restore_adam_and_rng_repeats_next_update(self):
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        model=torch.nn.Linear(3,2);opt=torch.optim.Adam(model.parameters(),lr=.01)
        def step():
            x=torch.randn(2,3)*random.random()+float(np.random.rand())
            opt.zero_grad();model(x).square().sum().backward();opt.step()
        step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            training.save_atomic(torch,dict(model=model.state_dict(),optimizer=opt.state_dict(),
                rng=training.rng_state(torch,np)),path,4096)
            step();expected=copy.deepcopy(model.state_dict())
            restored=torch.load(str(path),map_location='cpu')
            model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
            training.restore_rng(torch,np,restored['rng']);step()
            self.assertTrue(all(torch.equal(expected[k],v) for k,v in model.state_dict().items()))

    def test_storage_guard_keeps_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt';path.write_bytes(b'old checkpoint')
            with patch.object(training,'require_space',side_effect=RuntimeError('disk')):
                with self.assertRaises(RuntimeError):training.save_atomic(torch,{},path,4096)
            self.assertEqual(path.read_bytes(),b'old checkpoint')
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_partial_write_does_not_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt';path.write_bytes(b'old checkpoint')
            def fail(obj,target):
                Path(target).write_bytes(b'partial');raise OSError('write interrupted')
            with patch.object(torch,'save',side_effect=fail):
                with self.assertRaises(OSError):training.save_atomic(torch,{},path,4096)
            self.assertEqual(path.read_bytes(),b'old checkpoint')


if __name__=='__main__':unittest.main()

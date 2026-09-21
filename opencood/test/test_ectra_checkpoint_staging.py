import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from opencood.tools.run_ectra_ablation_suite import stage_checkpoint
from opencood.tools.train_utils import load_saved_model_diff


class CheckpointStagingTest(unittest.TestCase):
    def test_arbitrary_name_loads_exact_weights(self):
        for fallback in (False, True):
            with self.subTest(copy_fallback=fallback), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = nn.Linear(2, 1)
                checkpoint = root / 'diagnostic_bn13.pth'
                torch.save(source.state_dict(), checkpoint)
                folder = root / 'evaluation'
                folder.mkdir()
                if fallback:
                    with patch('os.link', side_effect=OSError('no hardlinks')):
                        target = stage_checkpoint(checkpoint, folder)
                else:
                    target = stage_checkpoint(checkpoint, folder)
                self.assertEqual(checkpoint.read_bytes(), target.read_bytes())
                _, loaded = load_saved_model_diff(
                    str(folder), nn.Linear(2, 1), require_checkpoint=True)
                for key, value in source.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.state_dict()[key]))

    def test_inference_rejects_unrecognized_name(self):
        with tempfile.TemporaryDirectory() as temp:
            model = nn.Linear(2, 1)
            torch.save(model.state_dict(), Path(temp) / 'diagnostic.pth')
            with self.assertRaisesRegex(FileNotFoundError, 'No loadable checkpoint'):
                load_saved_model_diff(temp, model, require_checkpoint=True)
            epoch, original = load_saved_model_diff(temp, model)
            self.assertEqual(epoch, 0)
            self.assertIs(original, model)


if __name__ == '__main__':
    unittest.main()

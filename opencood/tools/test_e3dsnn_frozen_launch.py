import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from opencood.tools.launch_e3dsnn_frozen_round import launch


class LaunchGuardTest(unittest.TestCase):
    def prepare(self, path):
        root = Path(path)
        (root / 'initializers').mkdir()
        (root / 'initializers/snn_single.pth').write_bytes(b'initializer')
        digest = hashlib.sha256(b'initializer').hexdigest()
        (root / 'stage1_summary.json').write_text(json.dumps({'passed': True}))
        for a in ('count4', 'relu'):
            (root / (a + '_stage1_pipeline.json')).write_text(json.dumps({'state': 'complete'}))
            (root / (a + '_frozen_smoke.json')).write_text(json.dumps({'passed': True, 'checkpoint_sha256': digest}))
        return root

    def test_busy_gpu_does_not_start_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.prepare(tmp)
            with patch('subprocess.check_output', return_value=b'22000, 2000, 80'), patch('subprocess.Popen') as popen:
                with self.assertRaises(RuntimeError): launch(root, root, 'count4', 6)
                popen.assert_not_called()
                self.assertFalse((root / 'count4_frozen30_launch.json').exists())

    def test_duplicate_and_changed_initializer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.prepare(tmp)
            (root / 'count4_frozen30_launch.json').write_text('{}')
            with self.assertRaises(FileExistsError): launch(root, root, 'count4', 6)
            (root / 'initializers/snn_single.pth').write_bytes(b'changed')
            with self.assertRaises(AssertionError): launch(root, root, 'count4', 6)


if __name__ == '__main__':
    unittest.main()

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from opencood.tools.launch_e3dsnn_aux_round import launch


class AuxLaunchTest(unittest.TestCase):
    def prepare(self, root, ap70):
        (root / 'frozen_final_summary.json').write_text(json.dumps({'passed': True,
            'models': {'count4': {'on': {'evaluation': {'content': {'ap': {'0.7': ap70}}}}}}}))
        (root / 'smoke_round.json').write_text(json.dumps({'passed': True}))
        repo = Path(__file__).resolve().parents[2]
        for a in ('count4', 'relu'):
            (root / (a + '_frozen30_pipeline.json')).write_text(json.dumps({'state': 'complete'}))
            for w in (0, 1):
                cfg = repo / 'opencood/hypes_yaml/dair-v2x/snn' / ('e3dsnn_pair_{}_aux{}_181.yaml'.format(a, w))
                (root / ('{}_aux{}_smoke.json'.format(a, w))).write_text(json.dumps({'passed': True,
                    'config_sha256': hashlib.sha256(cfg.read_bytes()).hexdigest()}))

    def test_wrong_branch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root, .54)
            with self.assertRaises(AssertionError): launch(root, root, root, 'count4', 1, 4)

    def test_busy_gpu_does_not_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.prepare(root, .56)
            with patch('subprocess.check_output', return_value=b'12000, 12000, 95'), patch('subprocess.Popen') as popen:
                with self.assertRaises(RuntimeError): launch(root, root, root, 'count4', 1, 4)
                popen.assert_not_called()
            self.assertFalse((root / 'count4_aux1_joint30_launch.json').exists())


if __name__ == '__main__':
    unittest.main()

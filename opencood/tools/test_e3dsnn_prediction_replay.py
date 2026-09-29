"""CPU checks for offline replay: geometry, empty frames and corrupted bundles."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from opencood.tools.replay_e3dsnn_predictions import replay


class PredictionReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        (self.folder / 'predictions').mkdir()
        box = np.asarray([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                          [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]], dtype=np.float32)
        above = box + np.asarray([0, 0, 3], dtype=np.float32)
        empty = np.empty((0, 8, 3), dtype=np.float32)
        frames = []
        # First prediction matches BEV only; the second is an exact 3D match.
        # Another frame has missed GT, and a third has neither detections nor GT.
        for i, (pred, score, gt) in enumerate([
                (np.stack([above, box]), [.9, .8], box[None]),
                (empty, [], box[None]), (empty, [], empty)]):
            path = self.folder / 'predictions' / ('%06d.npz' % i)
            np.savez_compressed(str(path), frame_id=np.asarray(str(i)), pred_boxes=pred,
                                scores=np.asarray(score, dtype=np.float32), gt_boxes=gt)
            frames.append(dict(index=i, frame_id=str(i), file=path.name,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               predicted_boxes=len(pred), gt_boxes=len(gt)))
        index_path = self.folder / 'predictions/index.json'
        index_path.write_text(json.dumps(dict(schema=1, frames=frames)))
        self.expected = dict(samples=3, predicted_boxes=2, gt_boxes=2,
                             ap={str(t): .5 for t in (.3, .5, .7)},
                             ap_3d={str(t): .25 for t in (.3, .5, .7)},
                             predictions=dict(frames=3, index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest()))
        self.write_expected()

    def write_expected(self):
        (self.folder / 'evaluation.json').write_text(json.dumps(self.expected))

    def test_known_ap_with_empty_frames(self):
        self.assertTrue(replay(self.folder)['exact_match'])

    def test_corrupted_prediction_is_rejected(self):
        path = self.folder / 'predictions/000000.npz'
        path.write_bytes(path.read_bytes() + b'corrupted')
        with self.assertRaises(AssertionError):
            replay(self.folder)

    def test_changed_index_is_rejected(self):
        path = self.folder / 'predictions/index.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(AssertionError):
            replay(self.folder)

    def test_wrong_reported_ap_is_rejected(self):
        self.expected['ap']['0.5'] = .51
        self.write_expected()
        with self.assertRaises(AssertionError):
            replay(self.folder)


if __name__ == '__main__':
    unittest.main()

import copy
import io
import unittest
import numpy as np
import torch
from torch import nn
from test_int_pyramid_cache import fixture
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_candidate_integer import CandidateIntegerDetector, CANDIDATE_PREFIXES


def source_and_quanta():
    source = ProtectedMultiscaleRefreshDetector(fixture()).eval()
    source.core.readout = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.Conv2d(128, 256, 1),
        nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1), nn.Conv2d(128, 256, 1),
        nn.BatchNorm2d(256), nn.ReLU()).eval()
    fine = {n: dict(quanta=[.125] * m.in_channels) for n, m in source.core.observation.named_modules() if isinstance(m, nn.Conv2d)}
    candidate = {n: dict(quanta=[.125] * m.in_channels) for n, m in source.core.named_modules()
                 if isinstance(m, nn.Conv2d) and n.startswith(CANDIDATE_PREFIXES)}
    return source, fine, [[.125] * c for c in (384, 128, 256, 128)], candidate


class CandidateIntegerTests(unittest.TestCase):
    def test_own_state_exact_skipped_refresh_reset_and_serialized_restore(self):
        source, fine, reader, candidate = source_and_quanta()
        models = [CandidateIntegerDetector(source, fine, reader, candidate, mode, True)
                  for mode in ('dense', 'static', 'delta')]
        states = [None] * 3
        schedule = [(i * 100000, 'a', False) for i in range(6)] + [(600000, 'b', False),
                    (700000, 'b', False), (1100000, 'b', False), (1200000, 'b', True)]
        with torch.no_grad():
            for step, (stamp, scene, explicit) in enumerate(schedule):
                x = torch.randn(1, 64, 8, 8)
                pose = np.eye(4); pose[0, 3] = step * .05
                meta = dict(timestamp_us=stamp, segment=scene, pose=pose, reset_before=explicit)
                before = states[2]
                outputs = []; infos = []
                for i, m in enumerate(models):
                    pred, states[i], info = m.step_spatial(x, states[i], meta)
                    outputs.append(pred); infos.append(info)
                for i in (1, 2):
                    self.assertTrue(all(torch.equal(outputs[0][k], outputs[i][k]) for k in outputs[0]))
                    self.assertTrue(all(torch.equal(states[0][k], states[i][k]) for k in source.state_keys))
                    for name, leaf in models[0].candidate_leaves().items():
                        other = models[i].candidate_leaves()[name]
                        self.assertEqual(leaf.audit is None, other.audit is None)
                        if leaf.audit is not None:
                            self.assertTrue(all(torch.equal(a, b) for a, b in zip(leaf.audit, other.audit)))
                self.assertEqual(states[0]['candidate_integer'], {})
                self.assertEqual(states[1]['candidate_integer'], {})
                expected_count = 5 if infos[2]['reset'] else 15
                self.assertEqual(len(states[2]['candidate_integer']), expected_count)
                if step == 4:
                    self.assertEqual(infos[2]['full_refresh'], 'period')
                    for name, pair in before['candidate_integer'].items():
                        if not name.startswith('fine_update.'):
                            self.assertIs(states[2]['candidate_integer'][name], pair)
                    stream = io.BytesIO(); torch.save(before, stream); stream.seek(0)
                    restored = torch.load(stream)
                    repeated, new, inf = models[2].step_spatial(x, restored, meta)
                    self.assertTrue(all(torch.equal(repeated[k], outputs[2][k]) for k in repeated))
                    for name, pair in new['candidate_integer'].items():
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pair, states[2]['candidate_integer'][name])))
                    self.assertEqual(inf['candidate_cache_bytes'], infos[2]['candidate_cache_bytes'])
                if step in (0, 6, 8, 9):
                    self.assertTrue(all(z['rebuild'] for z in infos[2]['candidate_integer'].values()))
                if step in (1, 7):
                    self.assertTrue(all(z['rebuild'] for n, z in infos[2]['candidate_integer'].items() if not n.startswith('fine_update.')))
                if step == 5:
                    self.assertTrue(all(not z['rebuild'] for z in infos[2]['candidate_integer'].values()))
                for m in models:
                    m.clear_audit()
                    self.assertTrue(all(v.previous is None and v.next_state is None and v.audit is None for v in m.candidate_leaves().values()))

    def test_explicit_mode_and_calibration_contract(self):
        source, fine, reader, candidate = source_and_quanta()
        with self.assertRaises(ValueError):
            CandidateIntegerDetector(source, fine, reader, candidate, 'auto')
        bad = copy.deepcopy(candidate); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):
            CandidateIntegerDetector(source, fine, reader, bad, 'static')


if __name__ == '__main__':
    unittest.main()

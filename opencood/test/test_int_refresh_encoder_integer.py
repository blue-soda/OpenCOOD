import io
import unittest
import numpy as np
import torch
from torch import nn
from test_int_candidate_integer import source_and_quanta
from opencood.models.sub_modules.int_refresh_encoder_integer import RefreshEncoderIntegerDetector, ENCODER_PREFIXES


def refresh_fixture():
    source, fine, reader, _ = source_and_quanta()
    calibration = {n: dict(quanta=[.125] * m.in_channels) for n, m in source.core.named_modules()
                   if isinstance(m, nn.Conv2d) and n.startswith(ENCODER_PREFIXES)}
    assert len(calibration) == 20
    return source, fine, reader, calibration


class RefreshEncoderIntegerTests(unittest.TestCase):
    def test_full_refresh_integer_equivalence_and_ordinary_reference_retention(self):
        source, fine, reader, calibration = refresh_fixture()
        models = [RefreshEncoderIntegerDetector(source, fine, reader, calibration, mode, True)
                  for mode in ('dense', 'static', 'delta')]
        states = [None] * 3
        schedule = [(i * 100000, 'a', False) for i in range(6)] + [(600000, 'b', False),
                    (700000, 'b', False), (1100000, 'b', False), (1200000, 'b', True)]
        with torch.no_grad():
            for step, (stamp, scene, explicit) in enumerate(schedule):
                x = torch.randn(1, 64, 8, 8)
                pose = np.eye(4); pose[0, 3] = step * .05
                meta = dict(timestamp_us=stamp, segment=scene, pose=pose, reset_before=explicit)
                before = states[2]; predictions = []; infos = []
                for i, model in enumerate(models):
                    pred, states[i], info = model.step_spatial(x, states[i], meta)
                    predictions.append(pred); infos.append(info)
                for i in (1, 2):
                    self.assertTrue(all(torch.equal(predictions[0][k], predictions[i][k]) for k in predictions[0]))
                    self.assertTrue(all(torch.equal(states[0][k], states[i][k]) for k in source.state_keys))
                    for name, reference in models[0].encoder_leaves().items():
                        other = models[i].encoder_leaves()[name]
                        self.assertEqual(reference.audit is None, other.audit is None)
                        if reference.audit is not None:
                            self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference.audit, other.audit)))
                self.assertEqual(states[0]['encoder_integer'], {})
                self.assertEqual(states[1]['encoder_integer'], {})
                self.assertEqual(len(states[2]['encoder_integer']), 20)
                executed = infos[2]['encoder_integer']
                self.assertEqual(len(executed), 20 if infos[2]['full_refresh'] else 0)
                if infos[2]['reset']:
                    self.assertTrue(all(z['rebuild'] for z in executed.values()))
                elif infos[2]['full_refresh']:
                    self.assertTrue(all(not z['rebuild'] for z in executed.values()))
                else:
                    for name, pair in before['encoder_integer'].items():
                        self.assertIs(states[2]['encoder_integer'][name], pair)
                if step == 4:
                    self.assertEqual(infos[2]['full_refresh'], 'period')
                    stream = io.BytesIO(); torch.save(before, stream); stream.seek(0)
                    restored = torch.load(stream)
                    repeated, new, info = models[2].step_spatial(x, restored, meta)
                    self.assertTrue(all(torch.equal(repeated[k], predictions[2][k]) for k in repeated))
                    for name, pair in new['encoder_integer'].items():
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pair, states[2]['encoder_integer'][name])))
                    self.assertEqual(info['encoder_cache_bytes'], infos[2]['encoder_cache_bytes'])
                for model in models:
                    model.clear_audit()
                    self.assertTrue(all(v.previous is None and v.next_state is None and v.audit is None for v in model.encoder_leaves().values()))

    def test_exact_layer_set_and_mode_required(self):
        source, fine, reader, calibration = refresh_fixture()
        with self.assertRaises(ValueError):
            RefreshEncoderIntegerDetector(source, fine, reader, calibration, 'auto')
        calibration.pop(next(iter(calibration)))
        with self.assertRaises(ValueError):
            RefreshEncoderIntegerDetector(source, fine, reader, calibration, 'static')


if __name__ == '__main__':
    unittest.main()

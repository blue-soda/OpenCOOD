"""Semantic tests against upstream INT and independently derived geometry."""
import ast
import copy
import math
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from opencood.models.sub_modules.int_feature_memory import (
    INTLayer, INTFusionModule, INTFeatureMemory, make_fusion_config, warp_bev)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def upstream_class():
    path = Path(__file__).resolve().parents[2] / 'third_party/INT/voxelnet.py.txt'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FusionModule')
    isolated = ast.Module(body=[node])  # Python 3.7 server runtime
    scope = dict(nn=nn, torch=torch, np=np, Sequential=INTLayer,
                 build_norm_layer=lambda cfg, c: ('bn', nn.BatchNorm2d(c, eps=cfg['eps'], momentum=cfg['momentum'])))
    exec(compile(isolated, str(path), 'exec'), scope)
    return scope['FusionModule']


class TestINTParity(unittest.TestCase):
    def test_upstream_concat_and_gru_forward_backward(self):
        torch.manual_seed(23)
        for mode in ['concat', 'gru']:
            cfg = make_fusion_config(mode, 4)
            port = INTFusionModule(copy.deepcopy(cfg)).eval()
            original = upstream_class()(AttrDict(copy.deepcopy(cfg))).eval()
            original.load_state_dict(port.state_dict(), strict=True)
            current = torch.randn(2, 4, 5, 7, requires_grad=True)
            reference_input = current.detach().clone().requires_grad_(True)
            past = torch.randn(2, 4 + int(mode == 'gru'), 5, 7)
            actual, state = port(current, [past.clone()], delta_t=.05)
            expected, reference_state = original(reference_input, [past.clone()])
            self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), mode)
            if state is not None:
                self.assertTrue(torch.equal(state, reference_state))
                self.assertFalse(state.requires_grad)
            actual.square().sum().backward()
            expected.square().sum().backward()
            self.assertTrue(torch.allclose(current.grad, reference_input.grad, atol=1e-7, rtol=1e-6))
            reference_parameters = dict(original.named_parameters())
            for name, p in port.named_parameters():
                q = reference_parameters[name]
                self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5))

    def test_real_delta_t_and_no_input_mutation(self):
        module = INTFusionModule(make_fusion_config('gru', 4)).eval()
        x, history = torch.ones(1, 4, 3, 3), torch.ones(1, 5, 3, 3)
        _, a = module(x, [history], delta_t=.1)
        _, b = module(x, [history], delta_t=.3)
        self.assertTrue(torch.equal(history, torch.ones_like(history)))
        self.assertTrue(torch.allclose(b[:, -1]-a[:, -1], torch.full_like(a[:, -1], .2)))

    def test_concat_identity_initialization(self):
        module = INTFusionModule(make_fusion_config('concat', 4)).eval()
        module.initialize_concat_identity()
        x = torch.rand(1, 4, 3, 7)
        y, _ = module(x, [torch.randn_like(x)])
        self.assertTrue(torch.allclose(y, x, atol=1e-6))


class TestINTGeometryAndState(unittest.TestCase):
    def setUp(self):
        self.region = [-4, -2, -1, 4, 2, 1]
        self.past = np.eye(4)
        self.value = torch.zeros(1, 1, 4, 8)
        self.value[0, 0, 2, 5] = 1  # world center (1.5, .5)

    def test_identity_and_translation(self):
        same, _ = warp_bev(self.value, self.past, self.past, self.region)
        self.assertTrue(torch.equal(same, self.value))
        pose = np.eye(4); pose[0, 3] = 1
        shifted, valid = warp_bev(self.value, self.past, pose, self.region)
        self.assertEqual(float(shifted[0, 0, 2, 4]), 1)
        self.assertEqual(float(shifted.sum()), 1)
        self.assertFalse(bool(valid[0, 0, 0, -1]))

    def test_rotation_on_non_square_extent(self):
        pose = np.eye(4)
        pose[:2, :2] = [[0, -1], [1, 0]]
        rotated, _ = warp_bev(self.value, self.past, pose, self.region)
        self.assertEqual(float(rotated[0, 0, 0, 4]), 1)  # current (.5, -1.5)
        self.assertEqual(float(rotated.sum()), 1)

    def test_scene_gap_reset_causality_and_detach(self):
        module = INTFeatureMemory('concat', 4, self.region).eval()
        x = torch.rand(1, 4, 4, 8, requires_grad=True)
        def meta(scene, stamp):
            return dict(segment=scene, timestamp_us=stamp, pose=np.eye(4).tolist())
        _, state, _ = module.step(x, None, meta('a', 1000000))
        self.assertFalse(state['value'].requires_grad)
        _, _, info = module.step(x, state, meta('a', 1100000))
        self.assertTrue(info['warp'])
        cross, _, info = module.step(x, state, meta('b', 1100000))
        fresh, _, _ = module.step(x, None, meta('b', 1100000))
        self.assertTrue(torch.equal(cross, fresh))
        self.assertEqual(info['reset'], 'scene')
        _, _, info = module.step(x, state, meta('a', 1500000))
        self.assertEqual(info['reset'], 'gap')
        with self.assertRaises(ValueError):
            module.step(x, state, meta('a', 900000))
        _, _, info = module.step(x, state, meta('a', 900000), reset=True)
        self.assertEqual(info['reset'], 'explicit')

    def test_learnable_history_path(self):
        module = INTFeatureMemory('concat', 4, self.region).eval()
        x = torch.rand(1, 4, 4, 8)
        meta = dict(segment='a', timestamp_us=1000000, pose=np.eye(4).tolist())
        _, state, _ = module.step(x, None, meta)
        meta['timestamp_us'] += 100000
        output, _, _ = module.step(x, state, meta)
        output.sum().backward()
        # History columns start at zero, but must receive a nonzero gradient.
        grad = module.fusion.post_fusion_conv[1].weight.grad[:, 4:]
        self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()

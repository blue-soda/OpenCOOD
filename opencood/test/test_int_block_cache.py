import unittest
from unittest.mock import patch
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.int_block_cache import BlockCachedBackbone, BlockCacheINT, block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev, INTFeatureMemory


class BlockCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(529)
        torch.set_num_threads(1)
        self.backbone = ResNetBEVBackbone(dict(layer_nums=[2, 2, 2], layer_strides=[2, 2, 2],
            num_filters=[4, 6, 8], upsample_strides=[1, 2, 4], num_upsample_filter=[4, 4, 4], inplanes=3)).eval()
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.bias.data.uniform_(-.2, .3)
                module.running_mean.uniform_(-.3, .3)
                module.running_var.uniform_(.5, 1.5)

    def test_dense_stages_exact_and_parameter_objects_unchanged(self):
        engine = BlockCachedBackbone(self.backbone)
        before = {k: v.clone() for k, v in self.backbone.state_dict().items()}
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 96)
            expected = self.backbone.resnet(x)
            stages, values, writes, stats = engine.run(x)
            for a, b in zip(expected, stages):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(self.backbone.decode_multiscale_feature(expected),
                                        self.backbone.decode_multiscale_feature(stages)))
            self.assertEqual(len(values), 6)
            self.assertTrue(all(bool(w.all()) for w in writes))
        self.assertIs(engine.backbone, self.backbone)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.backbone.state_dict().items()))

    def test_partial_blocks_match_masked_dense_recursively_and_preserve_unselected(self):
        engine = BlockCachedBackbone(self.backbone, tile=2, allow_fallback=False)
        with torch.no_grad():
            x = torch.randn(1, 3, 67, 99)
            _, cache, _, _ = engine.run(x)
            cache = [torch.randn_like(v) for v in cache]
            saved = [v.clone() for v in cache]
            masks = []
            for v in cache:
                mask = torch.zeros_like(v[:, :1], dtype=torch.bool)
                mask[..., 0, 0] = True
                mask[..., -1, -1] = True
                mask[..., v.shape[-2] // 2, v.shape[-1] // 2] = True
                masks.append(mask)
            # Each subsequent block consumes the preceding block's mixed output.
            sparse = engine.run(x, cache, masks, 'sparse')
            masked = engine.run(x, cache, masks, 'masked_dense')
            for a, b, old, write in zip(sparse[1], masked[1], saved, sparse[2]):
                self.assertTrue(torch.allclose(a, b, atol=3e-6, rtol=3e-5))
                self.assertTrue(torch.equal(a[~write.expand_as(a)], old[~write.expand_as(old)]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cache, saved)))
            for entry in engine.contract:
                self.assertEqual(entry['radius'], 3 if entry['stride'] == 2 else 2)

    def test_empty_masks_execute_no_residual_blocks(self):
        engine = BlockCachedBackbone(self.backbone, allow_fallback=False)
        calls = []
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 96)
            _, cache, _, _ = engine.run(x)
            hooks = [block.register_forward_hook(lambda *args: calls.append(1))
                     for i in range(3) for block in getattr(self.backbone.resnet, 'layer%d' % i)]
            try:
                result = engine.run(x + 5, cache, [torch.zeros_like(v[:, :1], dtype=torch.bool) for v in cache])
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertFalse(calls)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cache, result[1])))

    def test_reject_train_unknown_layout_and_nonlocal_normalization(self):
        engine = BlockCachedBackbone(self.backbone)
        with self.assertRaises(ValueError):
            engine.run(torch.zeros(1, 3, 32, 32))
        self.backbone.resnet.layer0[0].bn1 = nn.BatchNorm2d(4, track_running_stats=False)
        with self.assertRaises(ValueError):
            BlockCachedBackbone(self.backbone)

    def test_stride_phase_and_odd_feature_grid(self):
        bounds = [-12., -8., -1., 12., 8., 1.]
        for shape in [(16, 24), (17, 25)]:
            for scale in (2, 4, 8):
                output = tuple((v + scale - 1) // scale for v in shape)
                lattice = block_lidar_range(bounds, shape, output, scale)
                for lo, hi, axis in [(0, 3, 1), (1, 4, 0)]:
                    step = (bounds[hi] - bounds[lo]) / shape[axis]
                    self.assertAlmostEqual((lattice[hi] - lattice[lo]) / output[axis], scale * step)
                    self.assertAlmostEqual(lattice[lo] + .5 * scale * step, bounds[lo] + .5 * step)
        with self.assertRaises(ValueError):
            block_lidar_range(bounds, (16, 24), (5, 6), 4)

    def test_rotation_uses_convolution_lattice_centers(self):
        bounds = [-12., -8., -1., 12., 8., 1.]
        lattice = block_lidar_range(bounds, (16, 24), (4, 6), 4)
        value = torch.arange(24.).reshape(1, 1, 4, 6)
        past = torch.eye(4).numpy()
        current = past.copy()
        current[0, 0] = current[1, 1] = -1
        actual, valid = warp_bev(value, past, current, lattice)
        expected = torch.zeros_like(value)
        for y in range(4):
            for x in range(6):
                sx = int(round((-(-11.5 + 4 * x) + 11.5) / 4))
                sy = int(round((-(-7.5 + 4 * y) + 7.5) / 4))
                if 0 <= sx < 6 and 0 <= sy < 4:
                    expected[0, 0, y, x] = value[0, 0, sy, sx]
        self.assertTrue(torch.equal(actual, expected))

    def detector(self):
        class Pillar(nn.Module):
            def forward(self, batch):
                return dict(spatial_features=batch['voxel_features'])
        model = nn.Module()
        model.pillar_vfe, model.scatter = Pillar(), nn.Identity()
        model.backbone = self.backbone
        model.feature_memory = INTFeatureMemory('gru', 3, [-12, -8, -1, 12, 8, 1])
        model.cls_head, model.reg_head, model.dir_head = nn.Conv2d(12, 2, 1), nn.Conv2d(12, 14, 1), nn.Conv2d(12, 4, 1)
        model.shrink_flag, model.use_dir = False, True
        model.eval()
        return model, dict(model=dict(args=dict(lidar_range=[-12, -8, -1, 12, 8, 1])))

    def test_recursive_detector_sparse_masked_agreement_and_reuses_gru_alignment(self):
        model, cfg = self.detector()
        sparse = BlockCacheINT(model, cfg, threshold=.7, tile=2, allow_fallback=False)
        masked = BlockCacheINT(model, cfg, threshold=.7, tile=2, allow_fallback=False, execution='masked_dense')
        a_state = b_state = None
        with torch.no_grad():
            for i in range(4):
                pose = torch.eye(4).numpy()
                pose[0, 3] = i * .2
                meta = dict(segment='one', timestamp_us=i * 100000, pose=pose)
                data = dict(processed_lidar=dict(voxel_features=torch.randn(1, 3, 64, 96),
                                                voxel_coords=None, voxel_num_points=None))
                with patch('opencood.models.sub_modules.int_feature_memory.warp_bev', wraps=warp_bev) as warp:
                    a, a_state, info = sparse.step(data, a_state, meta)
                    self.assertEqual(warp.call_count, int(i > 0))
                b, b_state, _ = masked.step(data, b_state, meta)
                for key in a:
                    self.assertTrue(torch.allclose(a[key], b[key], atol=3e-6, rtol=3e-5))
                self.assertTrue(all(torch.allclose(x, y, atol=3e-6, rtol=3e-5)
                                    for x, y in zip(a_state['blocks'], b_state['blocks'])))

    def test_age_refresh_scene_reset_and_noncausal_rejection(self):
        model, cfg = self.detector()
        bridge = BlockCacheINT(model, cfg, threshold=1e6, max_age=.2, tile=2, allow_fallback=False)
        state = None
        data = dict(processed_lidar=dict(voxel_features=torch.randn(1, 3, 64, 96),
                                        voxel_coords=None, voxel_num_points=None))
        with torch.no_grad():
            for i in range(3):
                meta = dict(segment='one', timestamp_us=i * 100000, pose=torch.eye(4).numpy())
                _, state, info = bridge.step(data, state, meta)
                expected = 0. if i == 1 else 1.
                self.assertTrue(all(s['actual_write_fraction'] == expected for s in info['cache_blocks']))
            with self.assertRaises(ValueError):
                bridge.step(data, state, meta)
            meta = dict(meta, segment='two', timestamp_us=0)
            _, state, info = bridge.step(data, state, meta)
            self.assertEqual(info['reset'], 'scene')
            self.assertTrue(all(s['actual_write_fraction'] == 1 for s in info['cache_blocks']))


if __name__ == '__main__':
    unittest.main()

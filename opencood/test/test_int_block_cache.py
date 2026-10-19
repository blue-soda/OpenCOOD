import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.int_block_cache import BlockCachedBackbone


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


if __name__ == '__main__':
    unittest.main()

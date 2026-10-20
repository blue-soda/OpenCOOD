import unittest
import torch
from torch import nn
from opencood.models.sub_modules.codyntrust_resblock import BasicBlock
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.models.sub_modules.int_local_state_context import ContextLocalStateDetector, load_refinement_state
# unittest discover adds this test directory to sys.path. The repository also
# has opencood/test.py, so importing opencood.test would execute that old script.
from test_int_local_state import source_teacher


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(303)
        torch.set_num_threads(1)
        self.teacher = source_teacher()
        self.teacher.backbone.resnet.layer1.add_module('2', BasicBlock(256, 256))
        self.teacher.backbone.resnet.layer1.add_module('3', BasicBlock(256, 256))
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.cfg = dict(model=dict(args=dict(lidar_range=[-12,-8,-1,12,8,1])))

    def test_function_preserved_on_recursive_state_and_new_branch_learns(self):
        base = LocalStateDetector(self.teacher, self.cfg, 'recurrent')
        with torch.no_grad():
            base.update.input_projection.weight[:,128:384].fill_(.0001)
        model = ContextLocalStateDetector(self.teacher, self.cfg).train()
        load_refinement_state(model, base.state_dict())
        state = reference = None
        for i in range(3):
            x = torch.randn(1,64,16,24)
            meta = dict(timestamp_us=i*100000,segment='s',pose=torch.eye(4).numpy())
            expected, reference, _ = base.step_spatial(x, reference, meta)
            actual, state, _ = model.step_spatial(x, state, meta, detach_state=False)
            self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in expected))
        running = {k:v.clone() for k,v in model.state_dict().items() if 'running_' in k}
        sum(v.square().mean() for v in actual.values()).backward()
        for i in (2,3):
            self.assertGreater(float(model.update.blocks[i].bn2.weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in running.items()))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_nonidentity_context_radius_eight_crop_matches_dense(self):
        model = ContextLocalStateDetector(self.teacher, self.cfg)
        with torch.no_grad():
            for block in model.update.blocks:
                block.bn2.weight.fill_(.4)
                block.bn2.bias.fill_(.1)
            x = torch.randn(1,385,35,43)
            ref = model.update(x)
            old = torch.randn_like(ref)
            mask = torch.zeros_like(ref[:,:1], dtype=torch.bool)
            mask[...,0,0] = True
            mask[...,-1,-1] = True
            mask[...,17,21] = True
            actual, written, _ = refresh_tiles(model.update,x,old,mask,1,8,tile=2,allow_fallback=False)
            selected = written.expand_as(ref)
            self.assertTrue(torch.allclose(actual[selected],ref[selected],atol=1e-5,rtol=1e-4))
            self.assertTrue(torch.equal(actual[~selected],old[~selected]))

    def test_reject_incomplete_old_weights(self):
        base = LocalStateDetector(self.teacher,self.cfg)
        state = base.state_dict()
        del state['cls_head.weight']
        with self.assertRaises(ValueError):
            load_refinement_state(ContextLocalStateDetector(self.teacher,self.cfg), state)


if __name__ == '__main__':
    unittest.main()

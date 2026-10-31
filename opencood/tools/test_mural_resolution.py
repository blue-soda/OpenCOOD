"""Core invariants; run in the OpenCOOD environment with python -m unittest."""
import copy
import unittest
import torch
from torch import nn
from opencood.models.sub_modules.mural_resolution import install_resolution_bn, ResolutionBatchNorm, resolution_configs, mode_configs
from opencood.hypes_yaml.yaml_utils import load_yaml


class MuralCoreTest(unittest.TestCase):
    def test_initial_equivalence_and_shared_gradient(self):
        torch.manual_seed(303)
        original=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4)).eval()
        model=copy.deepcopy(original)
        conv=model[0]
        install_resolution_bn(model,3)
        x=torch.randn(2,3,8,8)
        for k in range(3):
            model[1].index=k
            self.assertTrue(torch.equal(original(x),model(x)))
        self.assertIs(model[0],conv)
        model.train(); model[1].index=1
        before=[b.running_mean.clone() for b in model[1].layers]
        model(x+3).square().sum().backward()
        self.assertTrue(torch.equal(before[0],model[1].layers[0].running_mean))
        self.assertFalse(torch.equal(before[1],model[1].layers[1].running_mean))
        self.assertTrue(torch.equal(before[2],model[1].layers[2].running_mean))
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNone(model[1].layers[0].weight.grad)
        self.assertIsNotNone(model[1].layers[1].weight.grad)

    def test_geometry(self):
        base=load_yaml('opencood/hypes_yaml/dair-v2x/repro/dair_stage1_codyntrust_single_wide.yaml')
        for config in resolution_configs(base):
            nx,ny,nz=config['model']['args']['point_pillar_scatter']['grid_size']
            vx,vy,vz=config['preprocess']['args']['voxel_size']
            self.assertAlmostEqual(nx*vx,201.6)
            self.assertAlmostEqual(ny*vy,80)
            self.assertEqual(nx%8,0);self.assertEqual(ny%8,0)
            self.assertEqual(config['postprocess']['anchor_args']['W'],nx)

    def test_fixed_grid_selection_and_joint_mode_guard(self):
        base=load_yaml('opencood/hypes_yaml/dair-v2x/repro/dair_stage1_codyntrust_single_wide.yaml')
        expected=resolution_configs(base)
        for index in range(3):
            self.assertEqual(mode_configs(base,'fixed',index),[expected[index]])
        self.assertEqual(mode_configs(base,'mural'),expected)
        self.assertEqual(mode_configs(base,'shared_bn'),expected)
        with self.assertRaises(ValueError): mode_configs(base,'mural',1)
        with self.assertRaises(ValueError): mode_configs(base,'fixed',3)


if __name__=='__main__':unittest.main()

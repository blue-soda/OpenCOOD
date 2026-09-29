"""Verify the ANN control only changes activations, not trainable parameters."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.e3dsnn_single import E3dsnnSingle
from opencood.models.e3dsnn_single_ann import E3dsnnSingleAnn
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    snn_cfg = load_yaml('opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml')
    ann_cfg = load_yaml('opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_ann.yaml')
    # Exclude parser-injected ndarray; compare source-independent YAML values.
    snn_compare, ann_compare = copy.deepcopy(snn_cfg), copy.deepcopy(ann_cfg)
    for cfg in (snn_compare, ann_compare):
        cfg.pop('name')
        cfg['model'].pop('core_method')
        cfg['model']['args'].pop('activation', None)
        cfg['model']['args']['grid_size'] = cfg['model']['args']['grid_size'].tolist()
    assert snn_compare == ann_compare, 'Non-activation experimental settings differ'
    torch.manual_seed(20260929)
    snn = E3dsnnSingle(snn_cfg['model']['args'])
    torch.manual_seed(20260929)
    ann = E3dsnnSingleAnn(ann_cfg['model']['args'])
    snn_state, ann_state = snn.state_dict(), ann.state_dict()
    assert snn_state.keys() == ann_state.keys()
    assert all(torch.equal(snn_state[key], ann_state[key]) for key in snn_state)
    counts = sum(isinstance(module, Multispike) for module in snn.modules())
    assert counts == ann.replaced_count_modules
    assert not any(isinstance(module, Multispike) for module in ann.modules())
    assert sum(isinstance(module, nn.ReLU) for module in ann.modules()) == counts
    report = {'config_equivalence_except_activation': True, 'identical_initial_state': True,
              'state_items': len(snn_state), 'replaced_count_modules': counts,
              'parameters': sum(p.numel() for p in ann.parameters()),
              'control': 'Count4 versus unbounded ReLU; not a rounding-only ablation'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

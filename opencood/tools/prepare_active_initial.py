"""Create a one-time active-initialization checkpoint for the depth variant.

Loads a source checkpoint into the active depth model, sets the appended
blocks' last BatchNorm scales to one, and saves the result. The saved file is
an ordinary state dict: training/eval then use the standard loader, so learned
values are never re-activated.
"""
import argparse
import hashlib
from pathlib import Path

import torch

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml_utils.load_yaml(args.config)
    model = train_utils.create_model(cfg)
    model.load_state_dict(torch.load(args.source, map_location='cpu'), strict=True)
    model.activate_appended_blocks()
    scales = []
    for stage in (3, 4):
        blocks = getattr(model.backbone_3d, 'conv' + str(stage)).blocks
        for block in blocks[-model.extra_depth:]:
            for branch in (block.block, block.voxel_block):
                scales.append(float(branch[-1].weight.mean()))
    torch.save(model.state_dict(), args.output)
    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print({'output': args.output, 'sha256': digest, 'activation_applied': model.activation_applied,
           'scale_means': scales})


if __name__ == '__main__':
    main()

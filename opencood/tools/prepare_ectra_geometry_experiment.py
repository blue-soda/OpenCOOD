"""Prepare matched geometry/history continuations, preserving source snapshots."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, os.getcwd())
from opencood.hypes_yaml.yaml_utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2103)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    snapshot = Path(args.snapshot).resolve()
    checkpoint = snapshot / 'net_epoch_bestval_at12.pth'
    config = load_yaml(str(snapshot / 'config.yaml'))
    if config['model']['core_method'] != 'point_pillar_where2comm_ectra':
        parser.error('Requires ECTRA best12 snapshot')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for mode in ('control', 'geometry', 'geometry_history'):
        folder = output / mode
        folder.mkdir()
        cfg = copy.deepcopy(config)
        cfg['name'] = 'ectra_' + mode + '_20260921'
        cfg['is_finetune'] = False
        cfg['train_params']['epoches'] = 12 + args.epochs
        cfg['optimizer']['lr'] = 0.0002
        cfg['lr_scheduler'] = dict(core_method='multistep', gamma=0.1,
                                   step_size=[13 + args.epochs])
        ectra = cfg['model']['args']['ectra']
        ectra['coordinate_mode'] = 'legacy' if mode == 'control' else 'collaborator_latest'
        ectra['require_ego_history'] = mode == 'geometry_history'
        cfg['ectra_ego_history'] = dict(enabled=mode == 'geometry_history', max_age_ms=100)
        (folder / 'config.yaml').write_text(yaml.dump(cfg), encoding='utf-8')
        os.link(str(checkpoint), str(folder / 'net_epoch12.pth'))
        seed_code = ('import random,numpy as np,torch,runpy;'
                     'random.seed({0});np.random.seed({0});torch.manual_seed({0});'
                     'torch.cuda.manual_seed_all({0});'
                     'runpy.run_path("opencood/tools/train.py",run_name="__main__")').format(args.seed)
        cmd = [sys.executable, '-u', '-c', seed_code, '--hypes_yaml', str(folder / 'config.yaml'),
               '--model_dir', str(folder), '--fusion_method', 'intermediate', '--two_stage', '1',
               '--num_workers', '8', '--skip_test']
        jobs.append(dict(mode=mode, folder=str(folder), command=cmd))
    manifest = dict(vars(args), source_checkpoint=str(checkpoint), sha256=sha, jobs=jobs,
                    evaluation='Use equal-budget final epoch; independent fresh AdamW states',
                    limitations='Continuation adaptation, not from-scratch causal evidence')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

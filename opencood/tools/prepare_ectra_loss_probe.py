"""Prepare matched short continuations; never modify a source run or checkpoint."""

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
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=1203)
    args = parser.parse_args()
    if args.epochs < 1 or args.epoch < 1 or args.lr <= 0:
        parser.error('epochs, epoch and lr must be positive')
    snapshot = Path(args.snapshot).resolve()
    checkpoint = snapshot / ('net_epoch_bestval_at%d.pth' % args.epoch)
    if not checkpoint.is_file():
        parser.error('Missing checkpoint: %s' % checkpoint)
    config = load_yaml(str(snapshot / 'config.yaml'))
    if config['model']['core_method'] != 'point_pillar_where2comm_ectra':
        parser.error('An ECTRA snapshot is required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    jobs = []
    for mode, loss_key in [('control', None),
                           ('no_motion_loss', 'ectra_motion_loss'),
                           ('no_roi_regularizer', 'ectra_roi_residual_loss')]:
        folder = output / mode
        folder.mkdir()
        params = copy.deepcopy(config)
        params['name'] = 'ectra_loss_probe_' + mode
        params['is_finetune'] = False
        params['train_params']['epoches'] = args.epoch + args.epochs
        params['optimizer']['lr'] = args.lr
        # Resume replays the scheduler, so place its milestone beyond this run.
        params['lr_scheduler'] = {'core_method': 'multistep', 'gamma': 0.1,
                                 'step_size': [args.epoch + args.epochs + 1]}
        if loss_key:
            params['loss']['args']['aux_weights'][loss_key] = 0.0
        (folder / 'config.yaml').write_text(yaml.dump(params), encoding='utf-8')
        os.link(str(checkpoint), str(folder / ('net_epoch%d.pth' % args.epoch)))
        seed_code = (
            'import random,numpy as np,torch,runpy;'
            'random.seed({0});np.random.seed({0});torch.manual_seed({0});'
            'torch.cuda.manual_seed_all({0});'
            'runpy.run_path("opencood/tools/train.py",run_name="__main__")'
        ).format(args.seed)
        command = [sys.executable, '-u', '-c', seed_code,
                   '--hypes_yaml', str(folder / 'config.yaml'),
                   '--model_dir', str(folder), '--fusion_method', 'intermediate',
                   '--two_stage', '1', '--skip_test']
        jobs.append({'mode': mode, 'folder': str(folder), 'command': command})
    manifest = dict(vars(args), source_checkpoint=str(checkpoint),
                    checkpoint_sha256=source_sha, jobs=jobs,
                    optimizer_state='fresh AdamW for every arm',
                    evaluation='Evaluate final equal-budget epoch, not each arm best loss',
                    limitation='Short continuation, not from-scratch or convergence evidence')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

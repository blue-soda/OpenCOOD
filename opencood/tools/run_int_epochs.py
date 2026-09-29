"""Full-segment INT training with epoch validation and restartable checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

from opencood.data_utils.int_ego_stream import sha256
from opencood.data_utils.int_sequence_order import segment_order
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.tools.run_int_ego import train, evaluate, write_json


def atomic_save(value, path):
    temporary = str(path)+'.tmp'
    torch.save(value, temporary)
    os.replace(temporary, str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True, help='Original P1 initialization')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['single', 'concat', 'gru'], required=True)
    p.add_argument('--config', default='opencood/hypes_yaml/dair-v2x/repro/dair_stage1_codyntrust_single_wide.yaml')
    p.add_argument('--seed', type=int, default=303)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-epochs', type=int, default=40)
    p.add_argument('--min-epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min-delta', type=float, default=.001)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 1 <= a.min_epochs <= a.max_epochs or a.patience < 1:
        p.error('Invalid epoch or patience limits')
    a.history_policy = 'aligned'
    a.train_steps = 0  # ignored by full_epoch=True
    a.clip_length = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    root = Path(a.output)
    if not a.resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root/'latest.pth').is_file():
        p.error('Resume requires an existing latest.pth')
    manifest = json.loads(Path(a.manifest).read_text())
    contract = dict(mode=a.mode, seed=a.seed, lr=a.lr, min_epochs=a.min_epochs,
                    patience=a.patience, min_delta=a.min_delta,
                    manifest_sha256=sha256(a.manifest), config_sha256=sha256(a.config),
                    initialization_sha256=sha256(a.checkpoint),
                    state_policy='whole-segment/per-frame-detach/v1')
    config = load_yaml(a.config)
    config['data_augment'] = []
    config['int_mode'] = a.mode
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = PointPillarINT(config['model']['args'], a.mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=.5, patience=2, threshold=a.min_delta,
        threshold_mode='abs', min_lr=1e-6)
    start, history, best, significant_best, stale = 1, [], -1., -1., 0
    if a.resume:
        saved = torch.load(str(root/'latest.pth'), map_location='cpu')
        if saved['contract'] != contract:
            raise ValueError('Resume contract differs; create a separate experiment')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, history = saved['epoch']+1, saved['history']
        best, significant_best, stale = saved['best'], saved['significant_best'], saved['stale']
        del saved
    else:
        model.load_single_checkpoint(torch.load(a.checkpoint, map_location='cpu'))
        write_json(root/'manifest.json', dict(contract=contract, arguments=vars(a),
                   code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                   python=sys.executable, torch=torch.__version__,
                   policy='Full segments, shuffled per epoch; same reset horizon as evaluation. BN statistics frozen, no augmentation, no BPTT beyond one scan.'))
        (root/'config.yaml').write_text(yaml.dump(config))
        write_json(root/'effective_runtime.json', dict(
            dataset='DairEgoStreamDataset', model='PointPillarINT', batch_size=1,
            mode=a.mode, artificial_delay=False, asynchronous_fusion=False,
            state_policy=contract['state_policy'], arguments=vars(a),
            optimizer=dict(type='AdamW', lr=a.lr, weight_decay=1e-4),
            scheduler=dict(type='ReduceLROnPlateau', metric='val_bev_ap70', factor=.5,
                           patience=2, threshold=a.min_delta, min_lr=1e-6),
            augmentation=[], bn_running_stats='frozen', tf32=False,
            note='Legacy fusion/train_params/optimizer/lr_scheduler fields in config.yaml are not executed by this dedicated runner.'))
    if start > a.max_epochs:
        p.error('No new epochs requested; increase --max-epochs to resume')
    write_json(root/'result.json', dict(status='RUNNING', next_epoch=start,
               max_epochs=a.max_epochs))
    completed = False
    for epoch in range(start, a.max_epochs+1):
        # Sampling and all RNG restart at epoch boundaries; no stream state crosses epochs.
        random.seed(a.seed+epoch); np.random.seed(a.seed+epoch)
        torch.manual_seed(a.seed+epoch); torch.cuda.manual_seed_all(a.seed+epoch)
        directory = root/('epoch_%03d' % epoch)
        if directory.exists():
            # Preserve evidence from an interrupted, not-yet-committed epoch.
            suffix = 1
            while (root/(directory.name+'_interrupted_%d' % suffix)).exists(): suffix += 1
            directory.rename(root/(directory.name+'_interrupted_%d' % suffix))
        directory.mkdir()
        if shutil.disk_usage(str(root)).free < 5*1024**3:
            raise RuntimeError('Less than 5 GiB free; stop before saving another epoch')
        print('EPOCH_START', a.mode, epoch, 'lr', optimizer.param_groups[0]['lr'], flush=True)
        order = segment_order(manifest['train'], a.seed+epoch)
        training = train(model, manifest['train'], config, a, directory,
                         optimizer=optimizer, order=order, full_epoch=True, save_optimizer=False)
        model.load_state_dict(torch.load(str(directory/'trained.pth'), map_location='cpu')['model'], strict=True)
        torch.cuda.reset_peak_memory_stats()
        evaluation = evaluate(model, manifest['val'], config, a, directory)
        if evaluation['scans'] != len(manifest['val']) or evaluation['labels'] != sum(r['supervised'] for r in manifest['val']):
            raise RuntimeError('Validation coverage mismatch')
        score = evaluation['metrics']['bev_ap70']
        if score > best:
            best = score
            write_json(root/'best.json', dict(epoch=epoch, bev_ap70=score,
                       checkpoint=str(directory/'trained.pth'), sha256=training['checkpoint_sha256']))
        if score > significant_best+a.min_delta:
            significant_best, stale = score, 0
        else:
            stale += 1
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step(score)
        item = dict(epoch=epoch, lr=lr_used, next_lr=optimizer.param_groups[0]['lr'],
                    training=training, evaluation=evaluation)
        history.append(item)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), contract=contract, epoch=epoch,
                    history=history, best=best, significant_best=significant_best, stale=stale), root/'latest.pth')
        write_json(root/'curve.json', history)
        write_json(directory/'result.json', dict(status='COMPLETED', **item))
        print('EPOCH_COMPLETE', a.mode, epoch, 'AP70', score, 'stale', stale, flush=True)
        if epoch >= a.min_epochs and stale >= a.patience and optimizer.param_groups[0]['lr'] <= a.lr/8:
            completed = True
            break
    write_json(root/'result.json', dict(status='PLATEAU_REACHED' if completed else 'EPOCH_LIMIT_REACHED',
               epochs=len(history), best=json.loads((root/'best.json').read_text()),
               caveat='Validation plateau is an operational stopping rule, not a proof of convergence or a held-out test result.'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if '--output' in sys.argv:
            failed_root = Path(sys.argv[sys.argv.index('--output')+1])
            if failed_root.is_dir():
                write_json(failed_root/'failure.json', dict(status='FAILED',
                           exception=type(exc).__name__, message=str(exc)))
        raise

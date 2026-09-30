"""Full coverage, real-frame TBPTT. Unlabelled scans retain temporal gradients."""
from collections import Counter
import json
import time

import numpy as np
import torch

from opencood.data_utils.int_ego_stream import sha256
from opencood.data_utils.int_tbptt import WindowOptimizer, window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.train_utils import to_device


def train_tbptt(model, order, config, args, directory, optimizer):
    from opencood.tools.run_int_ego import loader, freeze_bn_stats, write_json
    _, stream = loader(order, config, args.workers, args.seed)
    model.train()
    freeze_bn_stats(model)
    criterion = PointPillarDirLoss(config['loss']['args']).cuda()
    ends = window_ends(order, args.tbptt_steps)
    window = WindowOptimizer(model, optimizer)
    state, scans, labels, updates = None, 0, 0, 0
    resets, losses = Counter(), []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (directory/'train_frames.jsonl').open('w', buffering=1) as log:
        for index, sample in enumerate(stream):
            meta = sample['meta']
            if sample['ego'] is None:
                raise RuntimeError('Invalid frozen training frame %s: %s' % (meta['frame'], meta['invalid']))
            ego = to_device(sample['ego'], torch.device('cuda'))
            # Even context without a label must carry a gradient into future frames.
            prediction, state, info = model.step(ego, state, meta,
                reset=meta['reset_before'], detach_state=False)
            entry = dict(frame=meta['frame'], scene=meta['segment'],
                         supervised=meta['supervised'], **info)
            if meta['supervised']:
                loss = criterion(prediction, ego['label_dict'])
                window.add(loss)
                losses.append(float(loss.detach()))
                labels += 1
                entry['loss'] = losses[-1]
            if index in ends:
                state, update = window.finish(state)
                updates += int(update['optimizer_step'])
                entry.update(window_end=True, update=updates, **update)
            scans += 1
            if info['reset']:
                resets[info['reset']] += 1
            log.write(json.dumps(entry)+'\n')
            if scans % 64 == 0:
                print('TRAIN_TBPTT', args.mode, scans, '/', len(order), 'labels', labels,
                      'updates', updates, 'loss', losses[-1] if losses else None, flush=True)
    if scans != len(order) or labels != sum(r['supervised'] for r in order):
        raise RuntimeError('TBPTT coverage mismatch')
    if not losses or not updates:
        raise RuntimeError('No supervised TBPTT updates')
    checkpoint = directory/'trained.pth'
    torch.save(dict(model=model.state_dict(), updates=updates,
                    supervised_frames=labels, mode=args.mode, seed=args.seed), str(checkpoint))
    summary = dict(updates=updates, supervised_frames=labels, scans=scans,
                   windows=len(ends), tbptt_steps=args.tbptt_steps, resets=dict(resets),
                   loss_mean=float(np.mean(losses)),
                   first32_loss_mean=float(np.mean(losses[:32])), last32_loss_mean=float(np.mean(losses[-32:])),
                   seconds=time.monotonic()-started, checkpoint_sha256=sha256(checkpoint),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   protocol='Whole-segment state; detach at window boundary; mean labelled loss per window; one optimizer update after backward; unlabelled context differentiable; frozen BN statistics')
    write_json(directory/'train_summary.json', summary)
    return summary

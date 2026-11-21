"""Intervene on unsent middle BEV corrections at a fixed per-scan unit budget."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.calibrate_int_middle_budget import detector
from opencood.tools.check_int_deep_spike_stream import prepare_stream
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks
from opencood.utils import common_utils, eval_utils


QUANTUM = .25
ARM_NAMES = ('carry', 'drop_all', 'drop_integer', 'feedback_ann')
TRAIN_COUNTS = '6c095fdcedc8053935371dbcfb48e6eb9abdd6ecd276c72117bfb7e7006858f4'


def allocate(delta, budget, frame):
    flat = delta.flatten()
    indices = flat.nonzero().flatten()
    wanted = flat[indices].abs().long()
    demand = int(wanted.sum())
    if demand <= budget:
        return indices, flat[indices], demand
    # Unique 32-bit permutation within each magnitude, rotated for each scan.
    tie = ((indices + frame) * 2654435761) % (1 << 32)
    order = torch.argsort(wanted * (1 << 32) + tie, descending=True)
    wanted = wanted[order]
    indices = indices[order]
    room = (budget - (wanted.cumsum(0) - wanted)).clamp_min(0)
    count = torch.min(wanted, room)
    used = count > 0
    indices = indices[used]
    return indices, flat[indices].sign() * count[used], demand


class Sender:
    def __init__(self, mode, cap):
        self.mode = mode
        self.cap = cap
        self.sent = None

    def prepare(self, prev, meta, spatial, bounds, refresh):
        self.frame = int(meta['frame'])
        self.old = None
        if refresh is None:
            area = block_lidar_range(bounds, spatial.shape[-2:], self.sent.shape[-2:], 4)
            self.old = warp_bev(self.sent, prev['pose'], meta['pose'], area)[0]

    def __call__(self, value, written):
        target = torch.round(value / QUANTUM)
        if self.old is None:
            self.sent = target * QUANTUM
            self.record = dict(units=0, addresses=0, demand=0, withheld=0)
            return self.sent, self.sent
        old_level = self.old / QUANTUM
        idx, amplitude, demand = allocate(target - old_level, self.cap, self.frame)
        if self.mode == 'feedback_ann':
            # Analog residual state and one weighted message per selected address.
            self.sent = self.old.contiguous().clone()
            self.sent.view(-1).index_add_(0, idx, amplitude * QUANTUM)
            residual = value.double() - self.sent.double()
            stored = (self.sent.double() + residual).to(value.dtype)
        else:
            delta = torch.zeros_like(target, memory_format=torch.contiguous_format)
            delta.view(-1)[idx] = amplitude
            self.sent = (old_level + delta) * QUANTUM
            if self.mode == 'carry':
                stored = value
            elif self.mode == 'drop_integer':
                stored = self.sent + (value - target * QUANTUM)
            else:
                stored = self.sent
        units = int(amplitude.abs().sum(dtype=torch.float64))
        assert units == min(demand, self.cap)
        self.record = dict(units=units, addresses=len(idx), demand=demand, withheld=demand-units)
        return stored, self.sent


def score(post, ego, prediction, gt):
    boxes, scores = post.post_process(dict(ego=ego), dict(ego=prediction))
    stats = {t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)}
    for t in stats:
        eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, t)
    # Match globally in official score order before assigning GT distance bins.
    truth = gt.cpu().numpy()
    bins = np.digitize(np.linalg.norm(truth[:, :, :2].mean(1), axis=1), (30., 50.))
    counts = np.bincount(bins, minlength=3)
    hits = np.zeros(3, dtype=int)
    if boxes is not None:
        remaining = list(common_utils.convert_format(truth))
        ids = list(range(len(truth)))
        polys = common_utils.convert_format(boxes.cpu().numpy())
        for i in np.argsort(-scores.cpu().numpy()):
            if not ids:
                break
            overlap = common_utils.compute_iou(polys[i], remaining)
            pick = int(np.argmax(overlap))
            if overlap[pick] >= .7:
                hits[bins[ids.pop(pick)]] += 1
                remaining.pop(pick)
    assert sum(stats[.7]['tp']) == hits.sum() and counts.sum() == stats[.7]['gt']
    return dict(ap=stats, hits=hits.tolist(), targets=counts.tolist())


@torch.no_grad()
def run(inputs, scans, counts, out):
    assert digest(counts) == TRAIN_COUNTS
    train = [json.loads(line) for line in counts.read_text().splitlines()]
    demands = sorted(r['drop'][1] for r in train if not r['refresh'])
    cap = demands[(len(demands)-1)//10]
    source, cfg, rows, traces, starts, members, q = prepare_stream(inputs, scans)
    assert not ({r['scene'] for r in train} & {r['scene'] for r in rows})
    models = {arm:detector(source, q, 'carry', QUANTUM) for arm in ARM_NAMES}
    senders = {arm:Sender(arm, cap) for arm in models}
    for arm, model in models.items():
        model.coarse_interface = senders[arm]
    write_json(out/'contract.json', dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        members=members, sha256={k:digest(inputs/k) for k in ('plan.json','traces.json','quantum.json','config.yaml',
        'parent.pth','protected_light_final.pth','readout.pth')}, train_counts=TRAIN_COUNTS, budget=cap,
        quantum=QUANTUM, device=torch.cuda.get_device_name(), torch=torch.__version__,
        params=sum(p.numel() for p in source.parameters()), arms=ARM_NAMES, seed=303,
        refresh='dense rebuild and drop unsent residual in all arms', range_m=[30,50]))
    data, stream = loader(rows, cfg, 0, 303)
    n = 0
    with (out/'frames.jsonl').open('x', buffering=1) as raw:
        for sample, trace in zip(stream, traces):
            meta = sample['meta']
            assert meta['frame'] == trace['frame'] and meta['supervised'] == trace['supervised']
            if meta['frame'] in starts:
                states = dict.fromkeys(models)
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            request = local_request(trace_masks(trace, spatial.device))
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            record = dict(frame=meta['frame'], scene=meta['scene'], refresh=trace['refresh'], arms={})
            for arm, model in models.items():
                sender = senders[arm]
                refresh = model.refresh_reason(states[arm], meta)
                assert refresh == trace['refresh']
                sender.prepare(states[arm], meta, spatial, source.bounds, refresh)
                pred, state, info = model.step_spatial(spatial, states[arm], meta, request)
                assert info['full_refresh'] == refresh
                states[arm] = state
                if arm == 'carry':
                    reference = pred
                if arm == 'feedback_ann':
                    assert torch.equal(sender.sent, senders['carry'].sent)
                    assert torch.equal(state['coarse'], states['carry']['coarse'])
                    assert all(torch.equal(v, reference[k]) for k, v in pred.items())
                rec = dict(sender.record, deep=model.deep_projector.ledger)
                if gt is not None and arm != 'feedback_ann':
                    rec.update(score(data.post, ego, pred, gt))
                record['arms'][arm] = rec
            raw.write(json.dumps(record)+'\n')
            n += 1
    assert n == len(rows)
    write_json(out/'report.json', dict(status='COMPLETED', scans=n, feedback_exact=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('inputs','scans','counts','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.inputs, args.scans, args.counts, args.output)

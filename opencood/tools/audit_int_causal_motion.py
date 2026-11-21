"""Test local motion from two past shallow task states against a third, unseen state."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_demand import fine_support
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_fraction_history import HistoryExperiment, numeric_mask

OFFSETS = [(0,0)] + [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]


def descriptor(value):
    return value.abs().reshape(1,8,-1,*value.shape[-2:]).mean(2)


def estimate(older, last, valid):
    # Common support for all shifts, including border tiles, prevents padding wins.
    h,w = valid.shape[-2:]
    border = F.pad((~valid).float(), (1,1,1,1), value=1)
    valid = F.max_pool2d(border,3,1)==0
    pad = (0,-w%8,0,-h%8)
    def tile_sum(x):
        return F.avg_pool2d(F.pad(x,pad),8,8,divisor_override=1)
    n = tile_sum(valid.float())
    costs = torch.cat([tile_sum((last-torch.roll(older,shift,(-2,-1))).abs().mean(1,keepdim=True)*valid)
                       for shift in OFFSETS],1)
    best, choice = costs.min(1,keepdim=True)
    active = (n>=16) & (choice!=0) & (best<.85*costs[:,:1])
    choice = torch.where(active,choice,torch.zeros_like(choice))
    selected = choice.repeat_interleave(8,-2).repeat_interleave(8,-1)[:,:,:h,:w]
    selected = selected.masked_fill(~valid,0)
    stats = dict(eligible_tiles=int((n>=16).sum()), moved_tiles=int(active.sum()),
                 history_identity=float(costs[:,:1][active].sum()), history_motion=float(best[active].sum()))
    return selected, valid, stats


def synthetic():
    gen = torch.Generator().manual_seed(303)
    first = torch.rand((1,8,32,32),generator=gen)
    valid = torch.ones_like(first[:,:1],dtype=torch.bool)
    last = torch.roll(first,(0,1),(-2,-1))
    assert (estimate(first,last,valid)[0][:,:,1:-1,1:-1]==OFFSETS.index((0,1))).all()
    assert not estimate(first,first,valid)[0].any()


def next_errors(last, target, choices, valid, read_area):
    result = {}
    identity = (target-last).abs().mean(1,keepdim=True)
    forecast = identity.clone()
    for k,shift in enumerate(OFFSETS[1:],1):
        mask = choices==k
        error = (target-torch.roll(last,shift,(-2,-1))).abs().mean(1,keepdim=True)
        forecast[mask] = error[mask]
    for k,sel in dict(all=valid, moved=choices!=0, read=valid & read_area, moved_read=(choices!=0) & read_area).items():
        result[k] = dict(sites=int(sel.sum()),identity=float(identity[sel].double().sum()),
                         motion=float(forecast[sel].double().sum()))
    return result


def measure(previous, recent, current, area, read_area, source_key):
    oldest, valid0 = warp_bev(descriptor(previous[source_key]), previous['pose'],current['pose'],area)
    last, valid1 = warp_bev(recent[source_key],recent['pose'],current['pose'],area)
    choices, valid, result = estimate(oldest,descriptor(last),valid0 & valid1)
    result.update(next_errors(last,current[source_key],choices,valid,read_area))
    if source_key == 'observation':
        prior_task, task_valid = warp_bev(recent['value'],recent['pose'],current['pose'],area)
        assert torch.equal(valid1,task_valid)
        result['task'] = next_errors(prior_task,current['value'],choices,valid,read_area)
    return result


@torch.no_grad()
def run(a):
    synthetic()
    source,cfg,_,_,_,_,q = prepare_stream(a.inputs,a.scans)
    plan = read(a.training/'plan.json')
    book = read(a.book)
    report = read(a.training/'report.json')
    assert digest(a.book)==plan['book'] and digest(a.training/'plan.json')==report['plan']
    assert digest(a.training/'frames.jsonl')==report['frames']
    rows = [dict(r,supervised=False,label=None) for r in plan['rows']]
    assert all(digest(Path(r['pcd']))==plan['members'][r['pcd']] for r in rows)
    traces = [json.loads(s) for s in (a.training/'frames.jsonl').read_bytes().splitlines()]
    exp = HistoryExperiment(source,q,book)
    exp.models['carry'][1].cap = book['cap']//2
    captured = []
    source_key = 'observation' if a.observation else 'value'
    if a.observation:
        source.core.observation.register_forward_hook(lambda module, args, result: captured.append(result.detach().clone()))
    _,stream = loader(rows,cfg,0,303)
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace) in enumerate(zip(stream,traces)):
            meta = sample['meta']
            assert not meta['supervised'] and 'label_dict' not in sample['ego']
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                older = last = None
            spatial = spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            skip = numeric_mask(trace)
            _,state,_ = exp.step('carry',spatial,meta,last,trace,skip)
            if a.observation:
                assert len(captured)==1
                state['observation'] = captured.pop()
                assert state['observation'].shape==state['value'].shape
            row = dict(frame=meta['frame'],scene=meta['scene'],age=state['since_refresh'])
            if row['age']>=2:
                ratio = (rows[i]['timestamp_us']-rows[i-1]['timestamp_us'])/(rows[i-1]['timestamp_us']-rows[i-2]['timestamp_us'])
                row['interval_ratio'] = ratio
                if .8<=ratio<=1.25:
                    shape = state['value'].shape[-2:]
                    area = block_lidar_range(source.bounds,spatial.shape[-2:],shape,2)
                    row.update(measure(older,last,state,area,fine_support(~skip,shape),source_key))
            log.write(json.dumps(row)+'\n')
            older = {k: last[k] for k in ('pose',source_key)} if last else None
            last = state
    assert i+1==len(rows)==211
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=len(rows),labels=0,optimizer_steps=0,
        training=report['plan'],trace=report['frames'],book=digest(a.book),motion_source=source_key,
        code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    for k in ('inputs','scans','book','training','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--observation',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)


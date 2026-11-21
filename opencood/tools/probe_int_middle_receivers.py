"""Measure middle-state transmission and its support on the frozen R-319 clips."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_membrane import TaskStateInterface, score_arm
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


QUANTUM = .25


def middle_counts(stored, previous, spatial, meta, model, masks, info):
    level = torch.round(stored / QUANTUM)
    channels = level.shape[1]
    refresh = info['full_refresh'] is not None
    if refresh:
        delta = level
        forced = torch.ones_like(level[:, :1], dtype=torch.bool)
    else:
        bounds = block_lidar_range(model.source.bounds, spatial.shape[-2:], level.shape[-2:], 4)
        old = torch.round(previous['coarse'] / QUANTUM)
        old, _ = warp_bev(torch.cat((old, torch.ones_like(old[:, :1])), 1),
                          previous['pose'], meta['pose'], bounds)
        delta = level - old[:, :-1]
        forced = old[:, -1:] == 0
    changed = delta != 0
    points = changed.any(dim=1, keepdim=True)
    counts = dict(middle_sites=points.numel(), changed_sites=int(points.sum()),
                  forced_sites=int(forced.sum()), changed_or_forced_sites=int((points | forced).sum()),
                  changed_channels=int(changed.sum()), current_nonzero=int(level.count_nonzero()),
                  units=int(delta.abs().sum()), executed_deep_sites=0, affected_deep_sites=0,
                  affected_deep_channels=0, input_channels=channels)
    if refresh or not bool(masks['written'].any()):
        return counts
    # Use the same strided, halo-inclusive crops as the executable light encoder.
    _, jobs, work, _, _ = plan_host_coalesced_refresh(
        masks['written'], level.shape[-2:], model.stage.radius, model.tile)
    subset = [changed | forced] if jobs is None else [
        (changed | forced)[..., y0:y1, x0:x1] for _,_,_,_,y0,x0,y1,x1 in jobs]
    for crop in subset:
        support = F.max_pool2d(crop.float(), kernel_size=3, stride=2, padding=1).bool()
        active = support.any(dim=1)
        counts['executed_deep_sites'] += active.numel()
        counts['affected_deep_sites'] += int(active.sum())
        counts['affected_deep_channels'] += int(support.sum())
    assert counts['executed_deep_sites'] == work
    assert abs(info['stage_conv_ratio'] * masks['written'].numel() - work) < 1e-6
    return counts


@torch.no_grad()
def run(inputs, scans, reference, out):
    source, cfg, rows, traces, starts, members, q = prepare_stream(inputs, scans)
    ref = read(reference / 'contract.json')
    assert ref['members'] == members
    reference_stats = read(reference/'ap_records.json')['amplitude']
    arms = ('deep_only', 'middle_carry', 'middle_drop')
    models = {name:DeepBlockCache(source, 8, 'light', crop_first=True,
                    planner=plan_host_coalesced_refresh,
                    coarse_interface=None if name == 'deep_only' else TaskStateInterface(name.split('_')[1], QUANTUM),
                    deep_projector=DeepSpikeProjection(source.core.deblocks[2], q, 'amplitude'))
              for name in arms}
    data, stream = loader(rows, cfg, 0, 303)
    stats = {m:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for m in arms}
    middle = source.core.deblocks[1][0]
    dw, _, pw = source.core.deep_light
    assert middle.kernel_size == middle.stride == (2,2)
    assert dw.groups == dw.in_channels == pw.in_channels and pw.kernel_size == (1,1)
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               source=ref['source'], trace=digest(inputs/'traces.json'), reference=digest(reference/'ap_records.json'),
               members=members, quantum=QUANTUM, params=sum(p.numel() for p in source.parameters()),
               middle_shape=[middle.in_channels,middle.out_channels,2,2],deep_pw_outputs=pw.out_channels,
               device=torch.cuda.get_device_name(),torch=torch.__version__,prediction='dense decoded middle send; deep amplitude',
               geometry='3x3 stride-2 depthwise per actual crop; new coverage forces all channels; assumes ideal aligned downstream state'))
    with (out/'frames.jsonl').open('x',buffering=1) as f:
        for sample,trace in zip(stream,traces):
            meta = sample['meta']
            assert meta['frame'] == trace['frame']
            if meta['frame'] in starts:
                states = dict.fromkeys(arms)
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            masks = trace_masks(trace, spatial.device)
            rec = dict(frame=meta['frame'], scene=meta['scene'], refresh=trace['refresh'],
                       supervised=meta['supervised'], arms={})
            for name, model in models.items():
                previous = states[name]
                (pred, state, info), core = measured_step(model, spatial, previous, meta, local_request(masks))
                assert info['full_refresh'] == trace['refresh']
                if gt is not None:
                    score_arm(data.post,dict(ego=ego),pred,gt,stats[name])
                entry = dict(core_float_conv_flops=core, deep=model.deep_projector.ledger,
                             score_end=len(stats[name][.7]['score']), gt_end=stats[name][.7]['gt'])
                if name != 'deep_only':
                    entry['middle'] = middle_counts(state['coarse'],previous,spatial,meta,model,masks,info)
                    if name == 'middle_drop':
                        assert torch.equal(state['coarse'], torch.round(state['coarse']/QUANTUM)*QUANTUM)
                rec['arms'][name] = entry
                states[name] = state
            f.write(json.dumps(rec)+'\n')
    assert {str(t):s for t,s in stats['deep_only'].items()} == reference_stats
    assert sum(1 for _ in (out/'frames.jsonl').open()) == len(rows)
    write_json(out/'ap_records.json',stats)
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),deep_reference_exact=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('inputs','scans','reference','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.scans,args.reference,args.output)

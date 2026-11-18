"""Test same-shape crop batching on the fixed R-287 P5 trajectory."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from opencood.models.sub_modules import int_deep_block_cache as deep_cache
from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh, plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, WARMUP, fingerprint, fixed_trace
from opencood.tools.time_int_periodic_compact import exclusive


SLOTS = (1, 8)
MIN_GROUPABLE_ORDINARY = 8
ATOL, RTOL = 1e-6, 1e-5
STATE_KEYS = ('value', 'coarse', 'deep', 'deep_age')


class BatchSwitch:
    def __init__(self):
        self.slots = SLOTS[-1]
        self.audit = False
        self.last = None

    def __call__(self, module, x, cached, requested, **kwargs):
        result = coalesced_refresh(module, x, cached, requested, batch_size=self.slots, **kwargs)
        if self.audit:
            self.last = result[1]
        return result


class CapturePlan:
    last = None

    def __call__(self, mask, input_shape, radius, tile, stride):
        self.last = plan_host_coalesced_refresh(mask, input_shape, radius, tile, stride)
        return self.last


def layout_shapes(layout):
    if layout is None:
        return dict(kind='none', job_shapes=[], stage_calls={str(slots):0 for slots in SLOTS})
    jobs = layout[1]
    if jobs is None:
        return dict(kind='dense', job_shapes=[], stage_calls={str(slots):1 for slots in SLOTS})
    shapes = [(job[6] - job[4], job[7] - job[5]) for job in jobs]
    by_shape = Counter(shapes)
    return dict(kind='crops', job_shapes=shapes,
                shape_groups=[dict(height=h, width=w, jobs=count) for (h,w), count in sorted(by_shape.items())],
                stage_calls={str(slots):sum((count+slots-1)//slots for count in by_shape.values()) for slots in SLOTS})


def summarize_geometry(geometry):
    ordinary = [row for row in geometry if row['refresh'] is None]
    measured = [row for row in ordinary if row['index'] >= WARMUP]

    def counts(rows):
        calls = {str(slots):sum(r['layout']['stage_calls'][str(slots)] for r in rows) for slots in SLOTS}
        return dict(frames=len(rows), multi_job_frames=sum(len(r['layout']['job_shapes']) > 1 for r in rows),
            groupable_frames=sum(r['layout']['stage_calls']['8'] < r['layout']['stage_calls']['1'] for r in rows),
            stage_calls=calls, stage_calls_saved=calls['1']-calls['8'],
            crop_job_count_histogram=dict(Counter(len(r['layout']['job_shapes']) for r in rows)),
            dense_fallback_frames=sum(r['layout']['kind'] == 'dense' for r in rows))

    return dict(ordinary=counts(ordinary), post_warmup_ordinary=counts(measured))


def compare_tensors(ref, candidate, keys):
    rows = {}
    for key in keys:
        a, b = ref[key], candidate[key]
        delta = float((a-b).abs().max())
        rows[key] = dict(exact=torch.equal(a, b), close=torch.allclose(a,b,atol=ATOL,rtol=RTOL),
                         max_abs=delta if math.isfinite(delta) else None)
    return rows


def audit_fingerprint(pred, state):
    result = hashlib.sha256()
    for values in (pred, {k:state[k] for k in STATE_KEYS}):
        for key in sorted(values):
            result.update(values[key].cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def audit(source, cfg, rows, plan, masks, trace, out, switch):
    geometry = []
    captured = CapturePlan()
    cache = deep_cache.DeepBlockCache(source,TILE,'light',planner=captured,crop_first=True)
    batches = []

    def count_batch(module, inputs, output):
        n, _, height, width = inputs[0].shape
        batches.append((n, height, width))

    batch_hook = cache.stage.register_forward_hook(count_batch)
    switch.audit = True
    with (out/'geometry.jsonl').open('x',buffering=1) as raw:
        for _,ids,_,stream in clip_streams(plan,rows,cfg):
            states = dict.fromkeys(SLOTS)
            for index,sample in enumerate(stream):
                meta = sample['meta']
                assert ids[index] == meta['frame'] == trace[index]['frame']
                ego = to_device(sample['ego'],torch.device('cuda'))
                spatial = spatial_input(source,ego)
                predictions, costs, layouts, written, executed = {}, {}, {}, {}, {}
                for slots in SLOTS:
                    switch.slots, switch.last, captured.last = slots, None, None
                    batches.clear()
                    (predictions[slots],states[slots],info), costs[slots] = measured_step(
                        cache,spatial,states[slots],meta,masks[index])
                    assert info['full_refresh'] == trace[index]['arms']['local']['refresh']
                    layouts[slots] = captured.last
                    written[slots] = switch.last
                    executed[slots] = batches[:]
                refresh = info['full_refresh']
                geometry_exact = all(layouts[s] is None for s in SLOTS)
                write_exact = all(written[s] is None for s in SLOTS)
                if layouts[8] is not None:
                    geometry_exact = (layouts[1] is not None and layouts[1][1:] == layouts[8][1:]
                                      and torch.equal(layouts[1][0],layouts[8][0]))
                    write_exact = (all(written[s] is not None for s in SLOTS)
                                   and torch.equal(written[1],written[8])
                                   and torch.equal(written[8],layouts[8][0]))
                outputs = compare_tensors(predictions[8],predictions[1],predictions[8])
                state = compare_tensors(states[8],states[1],STATE_KEYS)
                core_exact = all(costs[s] == trace[index]['arms']['local']['core_conv_flops'] for s in SLOTS)
                archived = trace[index]['arms']['local']['fingerprint']
                anchor_exact = archived is None or fingerprint(predictions[8],states[8]) == archived
                layout = layout_shapes(layouts[8])
                geometry_exact &= all(len(executed[s]) == layout['stage_calls'][str(s)] for s in SLOTS)
                if layout['kind'] == 'crops':
                    geometry_exact &= all(Counter((h,w) for n,h,w in executed[s] for _ in range(n))
                                          == Counter(layout['job_shapes']) for s in SLOTS)
                row = dict(frame=ids[index], index=index, refresh=refresh, layout=layout,
                    geometry_exact=geometry_exact, write_mask_exact=write_exact, core_flops_exact=core_exact,
                    reference_checkpoint_exact=anchor_exact,
                    executed_batch_shapes=executed, core_flops=costs, prediction=outputs, state=state,
                    bitwise=all(r['exact'] for family in (outputs,state) for r in family.values()),
                    tolerance=all(r['close'] for family in (outputs,state) for r in family.values()),
                    checkpoint_fingerprint=audit_fingerprint(predictions[8],states[8]) if index%16==0 else None)
                geometry.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
                if (index+1)%16 == 0:
                    write_json(out/'status.json',dict(status='RUNNING',phase='geometry',scans=index+1))
    batch_hook.remove()
    switch.audit = False
    assert len(geometry) == len(trace)
    return geometry


def time_pair(source, cfg, rows, plan, masks, geometry, out, switch, uuid):
    cache = deep_cache.DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,crop_first=True)
    records, checks = [], 0
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(uuid)
            checks += 1
            for _,ids,_,stream in clip_streams(plan,rows,cfg):
                states = dict.fromkeys(SLOTS)
                for index,sample in enumerate(stream):
                    if index%16 == 0:
                        exclusive(uuid)
                        checks += 1
                    meta = sample['meta']
                    gold = geometry[index]
                    assert meta['frame'] == ids[index] == gold['frame']
                    ego = to_device(sample['ego'],torch.device('cuda'))
                    order = SLOTS if (index+repeat)%2 == 0 else tuple(reversed(SLOTS))
                    arms = {}
                    for slots in order:
                        switch.slots = slots
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        spatial = spatial_input(source,ego)
                        pred,states[slots],info = cache.step_spatial(spatial,states[slots],meta,masks[index])
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter()-start)*1000
                        assert info['full_refresh'] == gold['refresh']
                        if gold['checkpoint_fingerprint'] is not None:
                            assert audit_fingerprint(pred,states[slots]) == gold['checkpoint_fingerprint']
                        arms[str(slots)] = elapsed
                    row = dict(frame=ids[index], index=index, repeat=repeat, included=index>=WARMUP,
                               order=order, refresh=gold['refresh'], model_step_ms=arms,
                               grouped=gold['layout']['stage_calls']['1'] > gold['layout']['stage_calls']['8'])
                    records.append(row)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
                    if (index+1)%16 == 0:
                        write_json(out/'status.json',dict(status='RUNNING',phase='timing',pass_index=repeat,scans=index+1))
    exclusive(uuid)
    return records, checks+1


def summarize_time(records):
    used = [r for r in records if r['included']]
    result = {}
    for subset in ('all','ordinary','groupable'):
        if subset == 'all':
            rows = used
        elif subset == 'ordinary':
            rows = [r for r in used if r['refresh'] is None]
        else:
            rows = [r for r in used if r['refresh'] is None and r['grouped']]
        ms = {str(slots):[r['model_step_ms'][str(slots)] for r in rows] for slots in SLOTS}
        result[subset] = dict(samples_per_arm=len(rows), arms={name:dict(mean=mean(x),p95=float(np.percentile(x,95)))
                           for name,x in ms.items()}, paired_savings_ms_batch8=mean(a-b for a,b in zip(ms['1'],ms['8'])),
            paired_savings_ms_batch8_by_pass=[mean(r['model_step_ms']['1']-r['model_step_ms']['8']
                for r in rows if r['repeat']==i) for i in range(PASSES)])
    return result


def run(args,out):
    exclusive(args.uuid)
    plan = checked_plan(args.timing_plan)
    assert plan['split'] == 'val' and len(plan['clips']) == 1 and len(plan['clips'][0]) == 128
    plan_sha, trace_sha = digest(Path(args.timing_plan)), digest(Path(args.trace))
    ref = json.loads(Path(args.trace_contract).read_text())
    assert ref['plan_sha256'] == plan_sha and ref['source_sha256'] == SOURCE_SHA
    assert ref['reference_trace_sha256'] == trace_sha and 'crop' in ref['modes']
    assert (ref['period'],ref['tile'],ref['passes'],ref['warmup']) == (5,TILE,PASSES,WARMUP)
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    before = model_digest(source)
    scope = 'Pillar through detection heads, fixed optional masks; guard, online host planning, packing, warp and writeback included. Load, H2D, optional selection, postprocess, diagnostics and hooks excluded.'
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA,plan_sha256=plan_sha,reference_trace_sha256=trace_sha,
        reference_contract_sha256=digest(Path(args.trace_contract)),reference_code_commit=ref['code_commit'],
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,period=5,tile=TILE,
        batch_slots=SLOTS,passes=PASSES,warmup=WARMUP,minimum_post_warmup_groupable_ordinary=MIN_GROUPABLE_ORDINARY,
        exact_gate=True,diagnostic_tolerance=dict(atol=ATOL,rtol=RTOL),scope=scope))
    switch = BatchSwitch()
    with torch.no_grad():
        masks,trace = fixed_trace(deep_cache.DeepBlockCache(source,TILE,'light'),cfg,rows,plan,out)
        assert digest(out/'trace.jsonl')==trace_sha, 'R-287 reference trace must replay byte for byte'
        with patch.object(deep_cache,'coalesced_refresh',switch):
            geometry = audit(source,cfg,rows,plan,masks,trace,out,switch)
            distribution = summarize_geometry(geometry)
            keys = ('geometry_exact','write_mask_exact','core_flops_exact','reference_checkpoint_exact','bitwise','tolerance')
            passing = {key:sum(r[key] for r in geometry) for key in keys}
            reason = None
            if any(passing[key] != len(trace) for key in keys[:-1]):
                reason = 'equivalence_gate_failed'
            elif distribution['post_warmup_ordinary']['groupable_frames'] < MIN_GROUPABLE_ORDINARY:
                reason = 'too_few_same_shape_groups'
            timed, checks = ([],0) if reason else time_pair(source,cfg,rows,plan,masks,geometry,out,switch,args.uuid)
    assert model_digest(source)==before
    exclusive(args.uuid)
    ordinary = [r for r in geometry if r['index']>=WARMUP and r['refresh'] is None]
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=not reason,timing_skip_reason=reason,
        equivalence_frames=passing,totals=distribution,scans=len(trace),exclusive_checks=checks+2,
        timing=None if reason else summarize_time(timed),weights_unchanged=True,ap50=None,ap70=None,
        parameters=sum(p.numel() for p in source.parameters()),
        ordinary_core_flops_mean={str(slots):mean(r['core_flops'][slots] for r in ordinary) for slots in SLOTS}))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(trace),timing_valid=not reason,reason=reason))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','timing-plan','trace','trace-contract','uuid','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='precheck'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False))
        raise


if __name__ == '__main__':
    main()

"""Test a bounded merge of ragged crop jobs on the R-287 P5 trajectory."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch
from torch import nn

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.probe_int_deep_crop_batches import ATOL, RTOL, STATE_KEYS, audit_fingerprint, compare_tensors
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, WARMUP, fingerprint, fixed_trace
from opencood.tools.time_int_periodic_compact import exclusive


MODES = ('host_crop', 'merged_crop')
BATCH = 8
MAX_ADDED_PERCENT = 10
MIN_CHANGED_ORDINARY = 8


def crop_shape(job):
    return job[6] - job[4], job[7] - job[5]


def sites(job, stride):
    h,w = crop_shape(job)
    return ((h + stride - 1)//stride) * ((w + stride - 1)//stride)


def calls(jobs):
    return sum((n + BATCH - 1)//BATCH for n in Counter(map(crop_shape, jobs)).values())


def overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def bounds_pair(a, b):
    return tuple(min(x, y) if side in (0, 1, 4, 5) else max(x, y)
                 for side, (x,y) in enumerate(zip(a,b)))


def choose_merge(jobs, original, dense, stride):
    """Return the minimum added-work merge that removes a real batched call."""
    if jobs is None or len(jobs) < 2:
        return jobs, original, None
    old_calls = calls(jobs)
    best = None
    for i,j in combinations(range(len(jobs)), 2):
        joined = bounds_pair(jobs[i], jobs[j])
        remaining = [job for k,job in enumerate(jobs) if k not in (i,j)]
        if any(overlap(joined, job) for job in remaining):
            continue
        work = original + sites(joined,stride) - sites(jobs[i],stride) - sites(jobs[j],stride)
        if work * 100 > original * (100 + MAX_ADDED_PERCENT) or work > dense:
            continue
        candidate = jobs[:i] + [joined] + [job for k,job in enumerate(jobs[i+1:],start=i+1) if k != j]
        saved = old_calls - calls(candidate)
        if saved > 0:
            order = (work-original, -saved, i, j)
            if best is None or order < best[0]:
                best = (order, candidate, work, (i,j))
    return (jobs, original, None) if best is None else best[1:]


def merge_plan(mask, input_shape, radius, tile, stride):
    written,jobs,work,original,size = plan_host_coalesced_refresh(mask,input_shape,radius,tile,stride)
    updated,executed,_ = choose_merge(jobs,work,written.numel(),stride)
    return written,updated,executed,original,size


class EvidenceArm:
    def __init__(self, source, planner):
        self.logic = planner
        self.cache = DeepBlockCache(source,TILE,'light',planner=self.plan,deep_interface=self.observe,crop_first=True)
        self.batches = []
        self.hook = self.cache.stage.register_forward_hook(self.batch)
        self.layout = self.written = None

    def plan(self, mask, shape, radius, tile, stride):
        self.layout = self.logic(mask,shape,radius,tile,stride)
        return self.layout

    def observe(self, features, written):
        self.written = written
        return features,features

    def batch(self, module, inputs, output):
        n,_,h,w = inputs[0].shape
        self.batches.append((n,h,w))

    def reset(self):
        self.layout = self.written = None
        self.batches.clear()


def geometric_check(arm):
    if arm.layout is None:
        assert not arm.batches and (arm.written is None or not bool(arm.written.any()))
        return 0,None,0
    planned,jobs,work,_,_ = arm.layout
    assert torch.equal(planned,arm.written)
    if jobs is None:
        assert len(arm.batches)==1 and work==planned.numel()
        return work,None,1
    coverage = torch.zeros_like(planned,dtype=torch.int32)
    for y0,x0,y1,x1,iy0,ix0,iy1,ix1 in jobs:
        assert iy0%2==ix0%2==0 and 0<=iy0<iy1 and 0<=ix0<ix1
        assert 0<=y0<y1<=coverage.shape[-2] and 0<=x0<x1<=coverage.shape[-1]
        coverage[...,y0:y1,x0:x1] += 1
    assert bool((coverage <= 1).all()) and torch.equal((coverage>0)&planned,planned)
    assert work == sum(sites(j,2) for j in jobs)
    assert len(arm.batches) == calls(jobs)
    assert Counter((h,w) for n,h,w in arm.batches for _ in range(n)) == Counter(map(crop_shape,jobs))
    return work,jobs,len(arm.batches)


def audit(source,cfg,rows,plan,masks,trace,out):
    arms = {mode: EvidenceArm(source,planner) for mode,planner in zip(MODES,(plan_host_coalesced_refresh,merge_plan))}
    stage = arms[MODES[0]].cache.stage
    per_site = sum(2*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                   for m in stage.modules() if isinstance(m,nn.Conv2d))
    assert per_site > 0
    records = []
    with (out/'geometry.jsonl').open('x',buffering=1) as raw:
        for _,ids,_,stream in clip_streams(plan,rows,cfg):
            states = dict.fromkeys(MODES)
            for idx,sample in enumerate(stream):
                meta = sample['meta']
                gold = trace[idx]['arms']['local']
                assert ids[idx] == meta['frame'] == trace[idx]['frame']
                ego = to_device(sample['ego'],torch.device('cuda'))
                spatial = spatial_input(source,ego)
                predictions,flops,layouts = {},{},{}
                for mode, arm in arms.items():
                    arm.reset()
                    (predictions[mode],states[mode],info),flops[mode] = measured_step(
                        arm.cache,spatial,states[mode],meta,masks[idx])
                    assert info['full_refresh'] == gold['refresh']
                    layouts[mode] = geometric_check(arm)
                ref,new = MODES
                left,right = arms[ref].written,arms[new].written
                write_exact = ((left is right is None) or (left is not None and right is not None and torch.equal(left,right)))
                old_sites,old_jobs,old_calls = layouts[ref]
                new_sites,new_jobs,new_calls = layouts[new]
                changed = old_jobs != new_jobs
                if changed:
                    assert old_jobs is not None and new_jobs is not None
                    assert len(new_jobs) == len(old_jobs)-1 and new_calls < old_calls
                    assert new_sites*100 <= old_sites*(100+MAX_ADDED_PERCENT) and new_sites <= right.numel()
                else:
                    assert old_sites == new_sites and old_calls == new_calls
                expected = {ref:gold['core_conv_flops'],new:gold['core_conv_flops']+(new_sites-old_sites)*per_site}
                values = compare_tensors(predictions[ref],predictions[new],predictions[ref])
                recurrent = compare_tensors(states[ref],states[new],STATE_KEYS)
                checkpoint = gold['fingerprint']
                row = dict(frame=ids[idx],index=idx,refresh=gold['refresh'],changed=changed,write_mask_exact=write_exact,
                    planned_sites={ref:old_sites,new:new_sites},jobs={ref:old_jobs,new:new_jobs},
                    predicted_core_flops=expected,core_flops=flops,core_flops_exact=expected==flops,
                    stage_call_count={ref:old_calls,new:new_calls},executed_batch_shapes={m:a.batches[:] for m,a in arms.items()},
                    reference_checkpoint_exact=checkpoint is None or fingerprint(predictions[ref],states[ref])==checkpoint,
                    prediction=values,state=recurrent,
                    bitwise=all(r['exact'] for family in (values,recurrent) for r in family.values()),
                    tolerance=all(r['close'] for family in (values,recurrent) for r in family.values()),
                    audit_fingerprint=audit_fingerprint(predictions[ref],states[ref]) if idx%16==0 else None)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
                records.append(row)
                if (idx+1)%16==0:
                    write_json(out/'status.json',dict(status='RUNNING',phase='geometry',scans=idx+1))
    for arm in arms.values():
        arm.hook.remove()
    assert len(records)==len(trace)
    return records,per_site


def timed(source,cfg,rows,plan,masks,geometry,out,uuid):
    arms = {mode: DeepBlockCache(source,TILE,'light',planner=planner,crop_first=True)
            for mode,planner in zip(MODES,(plan_host_coalesced_refresh,merge_plan))}
    measurements,checks = [],0
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(uuid)
            checks += 1
            for _,ids,_,stream in clip_streams(plan,rows,cfg):
                states = dict.fromkeys(MODES)
                for idx,sample in enumerate(stream):
                    if idx%16==0:
                        exclusive(uuid)
                        checks += 1
                    meta = sample['meta']
                    reference = geometry[idx]
                    assert ids[idx]==meta['frame']==reference['frame']
                    ego = to_device(sample['ego'],torch.device('cuda'))
                    order = MODES if (idx+repeat)%2==0 else MODES[::-1]
                    times = {}
                    for mode in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        spatial = spatial_input(source,ego)
                        pred,states[mode],info = arms[mode].step_spatial(spatial,states[mode],meta,masks[idx])
                        torch.cuda.synchronize()
                        times[mode] = (time.perf_counter()-start)*1000
                        assert info['full_refresh']==reference['refresh']
                        if reference['audit_fingerprint'] is not None:
                            assert audit_fingerprint(pred,states[mode])==reference['audit_fingerprint']
                    row = dict(frame=ids[idx],index=idx,repeat=repeat,included=idx>=WARMUP,order=order,
                               refresh=reference['refresh'],changed=reference['changed'],model_step_ms=times)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
                    measurements.append(row)
                    if (idx+1)%16==0:
                        write_json(out/'status.json',dict(status='RUNNING',phase='timing',pass_index=repeat,scans=idx+1))
    exclusive(uuid)
    return measurements,checks+1


def summarize_timing(rows):
    all_rows = [r for r in rows if r['included']]
    ordinary = [r for r in all_rows if r['refresh'] is None]
    subsets = dict(all=all_rows,ordinary=ordinary,changed=[r for r in ordinary if r['changed']])
    result = {}
    for group,samples in subsets.items():
        arm = {m:[r['model_step_ms'][m] for r in samples] for m in MODES}
        deltas = [(r['repeat'],r['model_step_ms'][MODES[0]]-r['model_step_ms'][MODES[1]]) for r in samples]
        result[group] = dict(samples_per_arm=len(samples),
            arms={m:dict(mean=mean(x),p95=float(np.percentile(x,95))) for m,x in arm.items()},
            paired_savings_ms=mean(value for _,value in deltas),
            paired_savings_by_pass_ms=[mean(v for p,v in deltas if p==i) for i in range(PASSES)])
    return result


def run(args,out):
    exclusive(args.uuid)
    plan = checked_plan(args.timing_plan)
    assert plan['split']=='val' and len(plan['clips'])==1 and len(plan['clips'][0])==128
    plan_sha,trace_sha = digest(Path(args.timing_plan)),digest(Path(args.trace))
    prior = json.loads(Path(args.trace_contract).read_text())
    assert prior['plan_sha256']==plan_sha and prior['source_sha256']==SOURCE_SHA
    assert prior['reference_trace_sha256']==trace_sha and 'crop' in prior['modes']
    assert (prior['period'],prior['tile'],prior['passes'],prior['warmup'])==(5,TILE,PASSES,WARMUP)
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    original = model_digest(source)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA,plan_sha256=plan_sha,reference_trace_sha256=trace_sha,
        reference_contract_sha256=digest(Path(args.trace_contract)),reference_code_commit=prior['code_commit'],
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,period=5,tile=TILE,arms=MODES,
        max_added_sites_percent=MAX_ADDED_PERCENT,max_merges_per_frame=1,batch_size=BATCH,passes=PASSES,warmup=WARMUP,
        min_changed_postwarmup_ordinary=MIN_CHANGED_ORDINARY,equivalence='Bitwise outputs, four recurrent tensors, written masks, predicted and actual Conv/ConvTranspose counts. Tolerance diagnostic only.',
        tolerance=dict(atol=ATOL,rtol=RTOL),geometry='One minimum-added-work feasible bounding rectangle, no overlap with retained write jobs, no padding, and positive default-batch call savings.',
        timing_scope='Synchronized Pillar through heads, includes guard, online planning, pack, warp and writeback; excludes optional-mask selection, loader/H2D, postprocess and diagnostics.'))
    with torch.no_grad():
        masks,trace = fixed_trace(DeepBlockCache(source,TILE,'light'),cfg,rows,plan,out)
        assert digest(out/'trace.jsonl')==trace_sha, 'R-287 trace must replay byte for byte'
        geometry,per_site = audit(source,cfg,rows,plan,masks,trace,out)
        keys = ('write_mask_exact','core_flops_exact','reference_checkpoint_exact','bitwise','tolerance')
        passing = {k:sum(row[k] for row in geometry) for k in keys}
        included = [r for r in geometry if r['index']>=WARMUP]
        ordinary = [r for r in included if r['refresh'] is None]
        changed = sum(r['changed'] for r in ordinary)
        reason = None
        if any(passing[k]!=len(geometry) for k in keys[:-1]):
            reason = 'equivalence_gate_failed'
        elif changed < MIN_CHANGED_ORDINARY:
            reason = 'too_few_changed_ordinary'
        measures,checks = ([],0) if reason else timed(source,cfg,rows,plan,masks,geometry,out,args.uuid)
    assert model_digest(source)==original
    exclusive(args.uuid)
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=reason is None,timing_skip_reason=reason,
        equivalence_frames=passing,scans=len(geometry),changed_scans=sum(r['changed'] for r in geometry),
        postwarmup_ordinary=len(ordinary),changed_postwarmup_ordinary=changed,
        ordinary_stage_call_counts={m:sum(r['stage_call_count'][m] for r in ordinary) for m in MODES},
        ordinary_stage_site_counts={m:sum(r['planned_sites'][m] for r in ordinary) for m in MODES},
        postwarmup_core_flops_mean={m:mean(r['core_flops'][m] for r in included) for m in MODES},
        ordinary_core_flops_mean={m:mean(r['core_flops'][m] for r in ordinary) for m in MODES},
        core_flops_per_deep_output_site=per_site,model_weights_unchanged=True,exclusive_checks=checks+2,
        parameters=sum(p.numel() for p in source.parameters()),ap50=None,ap70=None,
        latency=None if reason else summarize_timing(measures)))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(geometry),timing_valid=reason is None,reason=reason))


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
        write_json(out/'status.json',dict(status='FAILED',timing_valid=False,error=repr(exc)))
        raise


if __name__=='__main__':
    main()

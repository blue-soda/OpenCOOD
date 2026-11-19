"""Same-card audit, latency and NVML energy of online deep-cache dispatch."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch
from torch import nn

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy, original_online_policy
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.measure_int_deep_phase_energy import BoardEnergy, idle, interval
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.profile_int_deep_phase_edges import ObservedPlanner, dense_plan
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, WARMUP, fingerprint
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.tools.train_int_deep_block_adapt import pack_mask


MODES = ('p5','crop_online','dispatch_online')
LOCAL = MODES[1:]
ENERGY_PASSES = 4
ENERGY_ORDER = tuple(permutations(MODES))
DEVICE = torch.device('cuda')


def build_models(source, planners):
    return dict(p5=source,**{
        name:DeepBlockCache(source,TILE,'light',planner=planners[name],crop_first=True,
                            deep_projector=DeepPhaseProjection(source.core.deblocks[2],indexed=True))
        for name in LOCAL})


def stage(cfg, rows, plan, uuid):
    clips = list(plan['clips'])
    entries = []
    for _,ids,_,stream in clip_streams(plan,rows,cfg):
        for i,sample in enumerate(stream):
            if i%16==0:
                exclusive(uuid)
            meta = sample['meta']
            assert meta['frame']==ids[i]
            ego = dict(processed_lidar=to_device(sample['ego']['processed_lidar'],DEVICE))
            entries.append((meta,ego))
    assert len(clips)==1 and len(entries)==len(clips[0])==128
    torch.cuda.synchronize()
    return entries


def equal_observations(a, b):
    if a is None or b is None:
        return a is b
    return all(x is y is None or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(a,b))


def quality_gate(source, entries, upstream, uuid, out):
    planners = dict(crop_online=ObservedPlanner(plan_host_coalesced_refresh),dispatch_online=ObservedPlanner(dense_plan))
    models = build_models(source,planners)
    models['original'] = DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,crop_first=True)
    policies = {n:ObservedPolicy(models[n],CachedResidualPolicy) for n in LOCAL}
    policies['original'] = ObservedPolicy(models['original'],original_online_policy)
    hooks = [models[n].stage.register_forward_pre_hook(planners[n].observe) for n in LOCAL]
    stage_cost = sum(2*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                     for m in models['original'].stage.modules() if isinstance(m,nn.Conv2d))
    proj = source.core.deblocks[2][0]
    proj_cost = 2*proj.in_channels*proj.out_channels*proj.kernel_size[0]*proj.kernel_size[1]
    states = dict.fromkeys(models)
    rows = []
    try:
        with (out/'equivalence.jsonl').open('x',buffering=1) as raw:
            for i,((meta,ego),gold) in enumerate(zip(entries,upstream)):
                if i%16==0:
                    exclusive(uuid)
                assert meta['frame']==gold['frame']
                spatial = spatial_input(source,ego)
                preds,costs,anchors,layouts = {},{},{},{}
                refresh = gold['arms']['p5']['refresh']
                for name,model in models.items():
                    policy = policies.get(name)
                    if policy is not None:
                        policy.observation = None
                    if name in LOCAL:
                        planners[name].reset()
                    (pred,states[name],info),cost = measured_step(model,spatial,states[name],meta,policy)
                    assert info['full_refresh']==refresh
                    if i%16==0:
                        anchors[name] = fingerprint(pred,states[name])
                    preds[name],costs[name] = pred,cost
                    if name in LOCAL:
                        layouts[name] = planners[name].record(states[name]['deep_age'],refresh)
                assert costs['p5']==gold['arms']['p5']['core_conv_flops']
                assert anchors.get('p5')==gold['arms']['p5']['fingerprint']
                obs = policies['original'].observation
                baseline = states['original']
                written = int((baseline['deep_age']==0).sum())
                sites = baseline['deep_age'].numel()
                for arm in LOCAL:
                    assert equal_observations(obs,policies[arm].observation)
                    assert all(torch.equal(preds['original'][k],preds[arm][k]) for k in preds['original'])
                    assert all(torch.equal(baseline[k],states[arm][k]) for k in (*source.state_keys,'deep_age'))
                    extra = layouts[arm]['work_sites']-layouts['crop_online']['work_sites']
                    assert costs[arm]==costs['original']-(sites-written)*proj_cost+stage_cost*extra
                assert torch.equal(states['crop_online']['deep_projection'],states['dispatch_online']['deep_projection'])
                valid,required,optional = (None,None,None) if obs is None else obs
                row = dict(frame=meta['frame'],index=i,refresh=refresh,core_conv_flops=costs,anchors=anchors,
                    written=written,invalid=0 if valid is None else int((~valid).sum()),
                    mandatory=pack_mask(required),optional=pack_mask(optional),
                    geometry_cache_hit=None if obs is None else policies['crop_online'].policy.record['geometry_cache_hit'],
                    fallback=layouts['crop_online']['jobs']!=layouts['dispatch_online']['jobs'],layouts=layouts)
                rows.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
    finally:
        for hook in hooks:
            hook.remove()
    assert len(rows)==len(entries)==len(upstream)==128
    return rows,stage_cost,proj_cost


def time_steps(source,models,entries,proof,uuid,out):
    measurements = []
    with (out/'timing.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(uuid)
            states = dict.fromkeys(MODES)
            policies = {n:CachedResidualPolicy(models[n]) for n in LOCAL}
            for i,((meta,ego),gold) in enumerate(zip(entries,proof)):
                if i%16==0:
                    exclusive(uuid)
                order = MODES[(i+repeat)%len(MODES):]+MODES[:(i+repeat)%len(MODES)]
                elapsed = {}
                for name in order:
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    spatial = spatial_input(source,ego)
                    pred,states[name],info = models[name].step_spatial(spatial,states[name],meta,policies.get(name))
                    torch.cuda.synchronize()
                    elapsed[name] = (time.perf_counter()-before)*1000
                    assert info['full_refresh']==gold['refresh']
                    if gold['anchors']:
                        assert fingerprint(pred,states[name])==gold['anchors'][name]
                row = dict(index=i,frame=meta['frame'],repeat=repeat,included=i>=WARMUP,order=order,
                           refresh=gold['refresh'],boundary=gold['invalid']>0,fallback=gold['fallback'],ms=elapsed)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
                measurements.append(row)
            write_json(out/'status.json',dict(status='RUNNING',phase='timing',passes=repeat+1))
    exclusive(uuid)
    return measurements


def replay(name,source,model,entries,proof=None):
    state = None
    policy = None if name=='p5' else CachedResidualPolicy(model)
    for i,(meta,ego) in enumerate(entries):
        spatial = spatial_input(source,ego)
        pred,state,info = model.step_spatial(spatial,state,meta,policy)
        torch.cuda.synchronize()
        if proof is not None:
            assert info['full_refresh']==proof[i]['refresh']
            if proof[i]['anchors']:
                assert fingerprint(pred,state)==proof[i]['anchors'][name]


def energy(meter,source,models,entries,proof,uuid,out):
    rows = []
    with (out/'board_intervals.jsonl').open('x',buffering=1) as raw:
        def record(row):
            rows.append(row)
            raw.write(json.dumps(row,allow_nan=False)+'\n')

        for block,order in enumerate(ENERGY_ORDER):
            base = dict(kind='idle',block=block,**idle(meter,uuid))
            record(base)
            for position,name in enumerate(order):
                exclusive(uuid)
                replay(name,source,models[name],entries,proof)
                before = meter.telemetry()
                first = meter.mark()
                for _ in range(ENERGY_PASSES):
                    replay(name,source,models[name],entries)
                last = meter.mark()
                after = meter.telemetry()
                exclusive(uuid)
                joules,seconds = interval(first,last)
                n = ENERGY_PASSES*len(entries)
                record(dict(kind='workload',block=block,position=position,arm=name,start=first,end=last,before=before,after=after,
                    joules=joules,seconds=seconds,scans=n,gross_j_per_scan=joules/n,board_w=joules/seconds,
                    ms_per_scan=1000*seconds/n))
                write_json(out/'status.json',dict(status='RUNNING',phase='board_intervals',block=block,position=position))
        record(dict(kind='idle',block=len(ENERGY_ORDER),**idle(meter,uuid)))
    return rows


def summary(timing,board,proof,parameters):
    kept = [r for r in timing if r['included']]
    normal = [r for r in kept if r['refresh'] is None]
    groups = dict(overall=kept,ordinary=normal,boundary=[r for r in normal if r['boundary']],
                  interior=[r for r in normal if not r['boundary']],fallback=[r for r in normal if r['fallback']])
    reports = {}
    for name in MODES:
        data = [r for r in board if r.get('arm')==name]
        stats = {}
        for group,frames in groups.items():
            if not frames:
                continue
            ms = [row['ms'][name] for row in frames]
            stats[group] = dict(samples=len(ms),mean_ms=mean(ms),p95_ms=float(np.percentile(ms,95)),
                core_conv_flops_per_scan=mean(proof[row['index']]['core_conv_flops'][name] for row in frames),
                pass_means=[mean(row['ms'][name] for row in frames if row['repeat']==p) for p in range(PASSES)])
        reports[name] = dict(parameters=parameters,ap50=None,ap70=None,timing=stats,gross_j_per_scan=mean(r['gross_j_per_scan'] for r in data),
            energy_scans=sum(r['scans'] for r in data),energy_ms_per_scan=mean(r['ms_per_scan'] for r in data),
            energy_mean_board_w=mean(r['board_w'] for r in data))
    return reports


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    meter = BoardEnergy(args.uuid)
    meter.mark()
    plan = checked_plan(args.plan)
    assert plan['split']=='val' and len(plan['clips'])==1 and len(plan['clips'][0])==128
    trace = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    previous = json.loads(Path(args.prior_contract).read_text())
    assert previous['reference_trace_sha256']==digest(Path(args.trace)) and previous['plan_sha256']==digest(Path(args.plan))
    assert previous['source_sha256']==SOURCE_SHA
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    weights = model_digest(source)
    paths = {key:Path(getattr(args,key)) for key in ('plan','trace','prior_contract')}
    hashes = {key:digest(path) for key,path in paths.items()}
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,source_sha256=SOURCE_SHA,inputs_sha256=hashes,
        modes=MODES,passes=PASSES,warmup=WARMUP,energy_order=ENERGY_ORDER,energy_passes=ENERGY_PASSES,
        scope='GPU-resident Pillar through detection heads, includes live residual, planner and same-mask dispatch. Independent model states. Loads, transfers and postprocess are outside timing. Energy uses a warm run and four homogeneous cold-scene replays per interval from synchronized NVML counter endpoints. AP not scored.'))
    entries = stage(cfg,rows,plan,args.uuid)
    proof,per_stage,per_projection = quality_gate(source,entries,trace,args.uuid,out)
    models = build_models(source,dict(crop_online=plan_host_coalesced_refresh,dispatch_online=dense_plan))
    times = time_steps(source,models,entries,proof,args.uuid,out)
    assert len(times)==len(proof)*PASSES
    work = energy(meter,source,models,entries,proof,args.uuid,out)
    assert model_digest(source)==weights and all(digest(paths[k])==v for k,v in hashes.items())
    ordinary = [r for r in proof if r['refresh'] is None]
    write_json(out/'report.json',dict(status='COMPLETED',verified_scans=len(proof),ordinary_scans=len(ordinary),
        boundary_scans=sum(r['invalid']>0 for r in ordinary),fallback_scans=sum(r['fallback'] for r in ordinary),
        geometry_cache_hits=sum(r['geometry_cache_hit'] for r in ordinary),stage_flops_per_site=per_stage,
        projection_flops_per_site=per_projection,metrics=summary(times,work,proof,sum(p.numel() for p in source.parameters()))))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(proof)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','trace','prior-contract','uuid','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()

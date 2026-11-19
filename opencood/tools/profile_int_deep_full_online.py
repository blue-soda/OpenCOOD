"""Measure strict-FP32 online deep-cache execution on the R-309 stream."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch

from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_full_online import (
    AGE, DENSE, PERIOD, SCANS, SEED, check_files, commit_segment,
    native_inputs, read, row_identity, segment_ranges, write,
)
from opencood.tools.eval_int_deep_block_online import ObservedPolicy
from opencood.tools.launch_int_memory import digest
from opencood.tools.measure_int_deep_phase_energy import BoardEnergy, idle, interval
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.profile_int_deep_composed_online import (
    COMPACT, DEVICE, build_models, cached_policy, equal_observations,
)
from opencood.tools.profile_int_deep_phase_edges import ObservedPlanner, dense_plan
from opencood.tools.run_int_ego import loader, to_device
from opencood.tools.time_int_periodic_compact import exclusive


MODES = ('p5', DENSE, COMPACT)
ORDERS = tuple(permutations(MODES))
PASSES = 3
PRECISION = dict(matmul_tf32=False, cudnn_tf32=False)


def batch_ranges(segments, cap):
    """Pack consecutive, complete segments up to the resident input limit."""
    assert segments and max(hi-lo for lo,hi in segments)<=cap
    batches = []
    start = end = segments[0][0]
    for lo,hi in segments:
        assert lo==end and lo<hi
        if hi-start>cap:
            batches.append((start,end))
            start = lo
        end = hi
    return [*batches,(start,end)]


def stage(cfg, rows, workers):
    _, stream = loader(rows,cfg,workers,SEED)
    entries = []
    for row,sample in zip(rows,stream):
        meta = sample['meta']
        assert row_identity(meta)==row_identity(row) and sample['ego'] is not None
        lidar = to_device(sample['ego']['processed_lidar'],DEVICE)
        entries.append((meta,dict(processed_lidar=lidar)))
    assert len(entries)==len(rows)
    torch.cuda.synchronize()
    return entries


def observed_policies(models):
    return {n:ObservedPolicy(models[n],cached_policy(n)) for n in MODES[1:]}


def audit(source, entries, gold, previous_segment):
    planners = {n:ObservedPlanner(dense_plan) for n in MODES[1:]}
    models = build_models(source,planners)
    states = dict.fromkeys(MODES,initial_state(previous_segment))
    segment = previous_segment
    checked = []
    with ExitStack() as stack:
        for name,planner in planners.items():
            stack.callback(models[name].stage.register_forward_pre_hook(planner.observe).remove)
        for (meta,ego),expected in zip(entries,gold):
            if meta['segment']!=segment:
                segment = meta['segment']
                policies = observed_policies(models)
            spatial = spatial_input(source,ego)
            preds,layouts = {},{}
            for name,model in models.items():
                policy = policies.get(name)
                if policy is not None:
                    policy.observation = None
                    planners[name].reset()
                (preds[name],states[name],info),cost = measured_step(model,spatial,states[name],meta,policy)
                prior = expected['arms'][DENSE if name==COMPACT else name]
                assert all(info[k]==prior[p] for k,p in (('full_refresh','refresh'),('reset','reset')))
                assert cost==prior['core_conv_flops'],(meta['frame'],name,cost)
                if policy is not None:
                    layouts[name] = planners[name].record(states[name]['deep_age'],info['full_refresh'])
                    assert json.loads(json.dumps(layouts[name]))==prior['layout'],(meta['frame'],name)
                    if info['full_refresh'] is None:
                        assert json.loads(json.dumps(policy.policy.record))==prior['selection'],(meta['frame'],name)
            old,new = (states[n] for n in (DENSE,COMPACT))
            assert all(torch.equal(old[k],new[k]) for k in (*source.state_keys,'deep_age','deep_projection'))
            assert all(torch.equal(preds[DENSE][k],preds[COMPACT][k]) for k in preds[DENSE])
            obs = policies[DENSE].observation
            assert equal_observations(obs,policies[COMPACT].observation)
            if obs is not None:
                valid,required,optional = obs
                written = required if optional is None else required|optional
                assert int((~valid).sum())==expected['invalid']
                assert written.flatten().nonzero().flatten().tolist()==expected['written_indices']
            else:
                assert expected['written_indices'] is None
            checked.append(expected['index'])
    return checked


def initial_state(previous_segment):
    return None if previous_segment is None else dict(scene=previous_segment)


def timing(source, models, entries, gold, previous_segment, uuid):
    samples = []
    for repeat in range(PASSES):
        exclusive(uuid)
        states = dict.fromkeys(MODES,initial_state(previous_segment))
        segment = previous_segment
        for (meta,ego),expected in zip(entries,gold):
            if meta['segment']!=segment:
                segment = meta['segment']
                policies = {n:cached_policy(n)(models[n]) for n in MODES[1:]}
            index = expected['index']
            base = MODES if index%2==0 else MODES[::-1]
            slot = (index+repeat)%len(MODES)
            order = base[slot:]+base[:slot]
            elapsed = {}
            for name in order:
                torch.cuda.synchronize()
                before = time.perf_counter()
                spatial = spatial_input(source,ego)
                _,states[name],info = models[name].step_spatial(spatial,states[name],meta,policies.get(name))
                torch.cuda.synchronize()
                elapsed[name] = (time.perf_counter()-before)*1000.
                assert info['full_refresh']==expected['refresh']
            samples.append(dict(index=index,repeat=repeat,order=order,ms=elapsed))
    return samples


def replay(name, source, model, entries, gold, previous_segment):
    state = initial_state(previous_segment)
    segment = previous_segment
    policy = None
    for (meta,ego),expected in zip(entries,gold):
        if meta['segment']!=segment:
            segment = meta['segment']
            policy = None if name=='p5' else cached_policy(name)(model)
        spatial = spatial_input(source,ego)
        _,state,info = model.step_spatial(spatial,state,meta,policy)
        torch.cuda.synchronize()
        assert info['full_refresh']==expected['refresh']


def energy(meter, source, models, entries, gold, previous_segment, batch, uuid):
    intervals = []
    for position,name in enumerate(ORDERS[batch%len(ORDERS)]):
        exclusive(uuid)
        before = meter.telemetry()
        first = meter.mark()
        replay(name,source,models[name],entries,gold,previous_segment)
        last = meter.mark()
        after = meter.telemetry()
        exclusive(uuid)
        joules,seconds = interval(first,last)
        intervals.append(dict(batch=batch,position=position,arm=name,scans=len(entries),
                              start=first,end=last,before=before,after=after,joules=joules,seconds=seconds))
    return intervals


def groups_for(row):
    if row['refresh'] is not None:
        return ('overall','refresh')
    hit = row['arms'][DENSE]['selection']['geometry_cache_hit']
    boundary = row['invalid']>0
    place,cache = ('edge' if boundary else 'interior'),('hit' if hit else 'miss')
    return 'overall','ordinary',place,'cache_'+cache,place+'_'+cache


def summarize(source, proof, times, intervals, batches):
    groups = {}
    for row in times:
        for label in groups_for(proof[row['index']]):
            groups.setdefault(label,[]).append(row)
    parameters = sum(p.numel() for p in source.parameters())
    result = {}
    for name in MODES:
        summaries = {}
        for key,rows in groups.items():
            times_ms = [r['ms'][name] for r in rows]
            summaries[key] = dict(samples=len(rows),mean_ms=mean(times_ms),p95_ms=float(np.percentile(times_ms,95)),
                                  pass_means=[mean(r['ms'][name] for r in rows if r['repeat']==p) for p in range(PASSES)])
        own = [r for r in intervals if r['arm']==name]
        scans = sum(r['scans'] for r in own)
        assert scans==len(proof)
        ledger = DENSE if name==COMPACT else name
        result[name] = dict(timing=summaries,energy_scans=scans,gross_j_per_scan=sum(r['joules'] for r in own)/scans,
                            energy_ms_per_scan=1000.*sum(r['seconds'] for r in own)/scans,
                            parameters=parameters,
                            core_conv_flops_mean=mean(r['arms'][ledger]['core_conv_flops'] for r in proof),
                            scored_ap50=None,scored_ap70=None)
    return dict(status='COMPLETED',batches=batches,scans=len(proof),passes=PASSES,
                refresh_scans=sum(r['refresh'] is not None for r in proof),metrics=result)


def load_batches(folder, ranges, identity):
    files = sorted(folder.glob('*.json'))
    assert len(files)<=len(ranges) and [p.stem for p in files]==['%04d'%i for i in range(len(files))]
    payloads = []
    for file,bounds in zip(files,ranges):
        item = read(file)
        payload = item['payload']
        assert sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()==item['sha256']
        lo,hi = bounds
        assert payload['contract']==identity and payload['bounds']==list(bounds)
        assert payload['verified']==list(range(lo,hi)) and len(payload['timing'])==(hi-lo)*PASSES
        assert [r['arm'] for r in payload['energy']]==list(ORDERS[len(payloads)%len(ORDERS)])
        payloads.append(payload)
    return payloads


@torch.no_grad()
def run(args, out):
    exclusive(args.uuid)
    meter = BoardEnergy(args.uuid)
    meter.mark()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = dict(plan=Path(args.plan),frames=Path(args.reference)/'frames.jsonl',
                 reference_contract=Path(args.reference)/'contract.json',reference_report=Path(args.reference)/'report.json')
    hashes = {n:digest(p) for n,p in paths.items()}
    assert hashes['frames']==args.reference_frames_sha256
    reference = read(paths['reference_contract'])['identity']
    report = read(paths['reference_report'])
    assert reference['precision']==PRECISION and reference['source_sha256']==SOURCE_SHA
    assert reference['plan_sha256']==hashes['plan'] and report['r170_p5_raw_ap_exact'] and report['layout_bitwise_frames']==SCANS
    plan = read(paths['plan'])
    proof = [json.loads(line) for line in paths['frames'].read_text().splitlines()]
    native,provenance = native_inputs(args.root)
    assert provenance['manifest_sha256']==plan['manifest_sha256'] and provenance['config_sha256']==plan['config_sha256']
    assert [row_identity(r) for r in native]==plan['ordered_stream']
    assert len(proof)==SCANS and all(p['index']==i and all(p[k]==native[i][k] for k in ('frame','scene','segment','supervised'))
                                   for i,p in enumerate(proof))
    ranges = batch_ranges(segment_ranges(native),args.batch_scans)
    if args.limit_batches is not None:
        assert 0<args.limit_batches<=len(ranges)
        ranges = ranges[:args.limit_batches]
    contract = dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=PRECISION,
                    input_sha256=hashes,source_sha256=SOURCE_SHA,torch=torch.__version__,driver=subprocess.check_output(
                        ['nvidia-smi','--query-gpu=driver_version','--id='+args.uuid,'--format=csv,noheader']).decode().strip(),
                    gpu=torch.cuda.get_device_name(),uuid=args.uuid,modes=list(MODES),energy_orders=[list(o) for o in ORDERS],
                    workers=args.workers,bounds=[list(r) for r in ranges],passes=PASSES,
                    scope='GPU-resident Pillar through detection heads, one full paired energy replay per arm per batch. '
                          'Each batch is preconditioned by a complete equivalence pass; all scans in the measured replays count. '
                          'Natural segment boundaries reset recurrent feature state and the candidate geometry policy. '
                          'No disk IO, voxelization, input transfer, AP or idle subtraction inside model intervals.')
    if args.resume:
        assert read(out/'contract.json')==contract
    else:
        write(out/'contract.json',contract)
    folder = out/'batches'
    folder.mkdir(exist_ok=args.resume)
    payloads = load_batches(folder,ranges,contract)
    check_files(plan,out)
    args.kind = 'geometry'
    source,cfg,rows,origin = build(args)
    source.period,source.max_age = PERIOD,AGE
    assert origin['split']=='val' and [row_identity(r) for r in rows]==plan['ordered_stream']
    weights = model_digest(source)
    models = build_models(source,{name:dense_plan for name in MODES[1:]})
    if not (out/'idle_before.json').exists():
        write(out/'idle_before.json',idle(meter,args.uuid))
    for batch,(lo,hi) in enumerate(ranges[len(payloads):],len(payloads)):
        old = rows[lo-1]['segment'] if lo else None
        def progress(phase):
            write(out/'status.json',dict(status='RUNNING',phase=phase,batch=batch,scans=lo,total=ranges[-1][1]))

        progress('stage')
        exclusive(args.uuid)
        torch.cuda.reset_peak_memory_stats()
        entries = stage(cfg,rows[lo:hi],args.workers)
        expected = proof[lo:hi]
        progress('audit')
        checked = audit(source,entries,expected,old)
        progress('timing')
        timing_rows = timing(source,models,entries,expected,old,args.uuid)
        progress('energy')
        intervals = energy(meter,source,models,entries,expected,old,batch,args.uuid)
        payload = dict(contract=contract,bounds=[lo,hi],verified=checked,timing=timing_rows,energy=intervals,
                       max_allocated_bytes=torch.cuda.max_memory_allocated())
        commit_segment(folder/('%04d.json'%batch),payload)
        payloads.append(payload)
        del entries
    write(out/'idle_after.json',idle(meter,args.uuid))
    assert model_digest(source)==weights and all(digest(p)==hashes[k] for k,p in paths.items())
    times = [row for payload in payloads for row in payload['timing']]
    intervals = [row for payload in payloads for row in payload['energy']]
    scanned = ranges[-1][1]
    assert all(p['verified']==list(range(*p['bounds'])) for p in payloads) and len(times)==scanned*PASSES
    result = summarize(source,proof[:scanned],times,intervals,len(ranges))
    result['max_allocated_bytes'] = max(p['max_allocated_bytes'] for p in payloads)
    write(out/'report.json',result)
    write(out/'status.json',dict(status='COMPLETED',scans=scanned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','reference','reference-frames-sha256','uuid','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--batch-scans',type=int,default=160)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit-batches',type=int)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()

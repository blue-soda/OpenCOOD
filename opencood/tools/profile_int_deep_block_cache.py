"""Fixed-trace phase diagnostics and same-card timing of deep-region planning."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, WARMUP, fingerprint, fixed_trace
from opencood.tools.time_int_periodic_compact import exclusive


class PhaseTimer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def phase(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        yield
        host_ms = (time.perf_counter()-before)*1000
        end.record()
        self.spans.append((name,start,end,host_ms))

    def summarize(self):
        totals = defaultdict(lambda: {'stream_ms':0., 'host_ms':0., 'calls':0})
        for name, start, end, host_ms in self.spans:
            totals[name]['stream_ms'] += start.elapsed_time(end)
            totals[name]['host_ms'] += host_ms
            totals[name]['calls'] += 1
        self.spans.clear()
        return dict(totals)


def assert_equal(left, right, keys):
    assert all(torch.equal(left[k],right[k]) for k in keys)


def prove_host(cache, host, source, cfg, rows, plan, masks, trace, out):
    verified = 0
    with (out/'host_equivalence.jsonl').open('x', buffering=1) as raw:
        for _, ids, _, stream in clip_streams(plan, rows, cfg):
            a, b = None, None
            for i, sample in enumerate(stream):
                meta = sample['meta']
                assert ids[i] == meta['frame'] == trace[i]['frame']
                ego = to_device(sample['ego'], torch.device('cuda'))
                spatial = spatial_input(source,ego)
                (ref,a,old), ca = measured_step(cache,spatial,a,meta,masks[i])
                (pred,b,new), cb = measured_step(host,spatial,b,meta,masks[i])
                assert old == new and ca == cb == trace[i]['arms']['local']['core_conv_flops']
                assert_equal(ref,pred,ref)
                assert_equal(a,b,(*source.state_keys,'deep_age'))
                raw.write(json.dumps(dict(frame=ids[i], refresh=old['full_refresh'], core_conv_flops=ca,
                    jobs=old.get('tiles'), state_and_prediction_exact=True, planning_stats_exact=True))+'\n')
                verified += 1
    return verified


def replay(source,cfg,rows,plan,masks,trace,out,uuid,modes,diagnostic):
    name = 'phase_frames.jsonl' if diagnostic else 'frames.jsonl'
    repeats = 1 if diagnostic else PASSES
    timers = {m:PhaseTimer() for m in modes} if diagnostic else {}
    phases = {m:timers[m].phase if diagnostic else nullcontext for m in modes}
    caches = {'local':DeepBlockCache(source,TILE,'light',phase=phases['local'])}
    if 'host' in modes:
        caches['host'] = DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,phase=phases['host'])
    measurements, checks = [], 0
    with (out/name).open('x', buffering=1) as raw:
        for repeat in range(repeats):
            exclusive(uuid)
            checks += 1
            for _, ids, _, stream in clip_streams(plan, rows, cfg):
                states = dict.fromkeys(modes)
                for i, sample in enumerate(stream):
                    if i%16 == 0:
                        exclusive(uuid)
                        checks += 1
                    meta = sample['meta']
                    assert ids[i] == meta['frame'] == trace[i]['frame']
                    ego = to_device(sample['ego'], torch.device('cuda'))
                    shift = (i+repeat) % len(modes)
                    order = modes[shift:] + modes[:shift]
                    arms = {}
                    for mode in order:
                        phase = phases[mode]
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        with phase('pillar'):
                            spatial = spatial_input(source,ego)
                        if mode == 'p5':
                            pred,states[mode],info = source.step_spatial(spatial,states[mode],meta,phase=phase)
                        else:
                            pred,states[mode],info = caches[mode].step_spatial(spatial,states[mode],meta,masks[i])
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter()-start)*1000
                        gold = trace[i]['arms']['p5' if mode=='p5' else 'local']
                        assert info['full_refresh'] == gold['refresh']
                        if gold['fingerprint'] is not None:
                            assert fingerprint(pred,states[mode]) == gold['fingerprint']
                        arms[mode] = dict(model_step_ms=elapsed, refresh=info['full_refresh'],
                                          core_conv_flops=gold['core_conv_flops'])
                        if diagnostic:
                            arms[mode]['phases'] = timers[mode].summarize()
                    record = dict(frame=ids[i],repeat=repeat,index=i,included=i>=WARMUP,order=order,arms=arms)
                    raw.write(json.dumps(record,allow_nan=False)+'\n')
                    measurements.append(record)
                    if (i+1)%16 == 0:
                        write_json(out/'status.json',dict(status='RUNNING',phase=name,pass_index=repeat,scans=i+1))
    exclusive(uuid)
    return measurements, checks+1


def summarize(plain, diagnostic, modes, parameters):
    result, breakdown = {}, {}
    for mode in modes:
        arm = [r['arms'][mode] for r in plain if r['included']]
        samples = [r['model_step_ms'] for r in arm]
        ordinary = [r['model_step_ms'] for r in arm if r['refresh'] is None]
        result[mode] = dict(parameters=parameters,ap50=None,ap70=None,samples=len(samples),
            model_step_ms_mean=mean(samples),model_step_ms_p95=float(np.percentile(samples,95)),
            ordinary_step_ms_mean=mean(ordinary),ordinary_step_ms_p95=float(np.percentile(ordinary,95)),
            core_conv_flops_mean=mean(r['core_conv_flops'] for r in arm),
            pass_means=[mean(r['arms'][mode]['model_step_ms'] for r in plain if r['included'] and r['repeat']==i)
                        for i in range(PASSES)],
            pass_ordinary_means=[mean(r['arms'][mode]['model_step_ms'] for r in plain
                                 if r['included'] and r['repeat']==i and r['arms'][mode]['refresh'] is None)
                                 for i in range(PASSES)])
        segments = [r['arms'][mode]['phases'] for r in diagnostic
                    if r['included'] and r['arms'][mode]['refresh'] is None]
        breakdown[mode] = {
            key:{metric:mean(row.get(key,{}).get(metric,0.) for row in segments)
                 for metric in ('stream_ms','host_ms','calls')}
            for key in sorted(set().union(*segments))
        }
    return result, breakdown


def worker(args,out):
    exclusive(args.uuid)
    plan = checked_plan(args.timing_plan)
    assert plan['split']=='val' and len(plan['clips'])==1 and len(plan['clips'][0])==128
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    before = model_digest(source)
    cache = DeepBlockCache(source,TILE,'light')
    modes = ('p5','local','host') if args.compare_host else ('p5','local')
    scope = 'Pillar through detection heads, fixed optional masks; includes guard, online region planning, packing, warp and writeback. Excludes loaders, H2D, postprocess and off-timestep optional selection.'
    diagnostic_scope = 'Separate diagnostic pass: host wall blocks may wait on earlier kernels, CUDA event intervals include host gaps. These readings are not isolated kernel costs or additive across host and stream. The model latency pass has no active timers, hooks or internal synchronization beyond the model itself.'
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA, plan_sha256=digest(Path(args.timing_plan)),reference_trace_sha256=digest(Path(args.trace)),
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,period=5,tile=TILE,
        passes=PASSES,warmup=WARMUP,modes=modes,scope=scope,diagnostic_scope=diagnostic_scope))
    with torch.no_grad():
        masks, trace = fixed_trace(cache,cfg,rows,plan,out)
        assert digest(out/'trace.jsonl')==digest(Path(args.trace)), 'R-276 trace must replay byte for byte'
        equivalent = 0
        if args.compare_host:
            host = DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh)
            equivalent = prove_host(cache,host,source,cfg,rows,plan,masks,trace,out)
            assert equivalent==len(trace)
        plain,pchecks = replay(source,cfg,rows,plan,masks,trace,out,args.uuid,modes,False)
        diagnostic,dchecks = replay(source,cfg,rows,plan,masks,trace,out,args.uuid,modes,True)
    assert model_digest(source)==before
    metrics, phases = summarize(plain,diagnostic,modes,sum(p.numel() for p in source.parameters()))
    write_json(out/'report.json',dict(status='COMPLETED',timing_valid=True,weights_unchanged=True,
        trace_identity=True,metrics=metrics,ordinary_phase_means=phases,host_equivalence_frames=equivalent,
        scopes=dict(model=scope,phases=diagnostic_scope),exclusive_checks=pchecks+dchecks))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(trace),phases=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','timing-plan','trace','uuid','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--compare-host',action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='precheck'))
    try:
        worker(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False))
        raise


if __name__ == '__main__':
    main()

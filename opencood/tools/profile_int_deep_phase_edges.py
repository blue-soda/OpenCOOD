"""Compare real same-write deep execution layouts on the fixed boundary trace."""
import argparse
from functools import reduce
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch
from torch import nn

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.probe_int_deep_crop_merge import bounds_pair, calls, merge_plan, sites
from opencood.tools.profile_int_deep_block_cache import PROJECTION_TOLERANCE, replay
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, WARMUP, fingerprint, fixed_trace
from opencood.tools.time_int_periodic_compact import exclusive


def joint_plan(mask, shape, radius, tile, stride, entire=False):
    written,jobs,work,original,size = plan_host_coalesced_refresh(mask,shape,radius,tile,stride)
    if jobs is None or calls(jobs)<2:
        return written,jobs,work,original,size
    full = written.numel()
    if entire:
        return written,None,full,original,None
    joined = reduce(bounds_pair,jobs)
    joined_work = sites(joined,stride)
    if joined_work>=full:
        return written,jobs,work,original,size
    return written,[joined],joined_work,original,size


def dense_plan(mask, shape, radius, tile, stride):
    return joint_plan(mask,shape,radius,tile,stride,entire=True)


PLANNERS = {'crop_index':plan_host_coalesced_refresh, 'bounded':merge_plan,
            'envelope':joint_plan, 'dense':dense_plan}
MODES = ('p5',*PLANNERS)


class ObservedPlanner:
    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def __call__(self, *args):
        self.layout = self.planner(*args)
        return self.layout

    def observe(self, module, args):
        n,_,h,w = args[0].shape
        self.shapes.append((n,h,w))

    def reset(self):
        self.layout = None
        self.shapes = []

    def record(self, age, refresh):
        if self.layout is None:
            assert not self.shapes
            return dict(jobs=[],work_sites=0,calls=0,batches=[])
        written,jobs,area,_,_ = self.layout
        assert refresh is None and torch.equal(written,age==0)
        assert area==sum(n*((h+1)//2)*((w+1)//2) for n,h,w in self.shapes)
        assert len(self.shapes)==(1 if jobs is None else calls(jobs))
        return dict(jobs=jobs,work_sites=area,calls=len(self.shapes),batches=self.shapes)


def equivalence(source,cfg,rows,plan,masks,trace,prior,out,uuid):
    observers = {name:ObservedPlanner(planner) for name,planner in PLANNERS.items()}
    arms = {name:DeepBlockCache(source,TILE,'light',planner=observer,crop_first=True,
                                deep_projector=DeepPhaseProjection(source.core.deblocks[2],indexed=True))
            for name,observer in observers.items()}
    hooks = [arm.stage.register_forward_pre_hook(observers[name].observe) for name,arm in arms.items()]
    flops_per_site = sum(2*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                         for m in arms['crop_index'].stage.modules() if isinstance(m,nn.Conv2d))
    records = []
    try:
        with (out/'equivalence.jsonl').open('x',buffering=1) as raw:
            for _,ids,_,stream in clip_streams(plan,rows,cfg):
                states = dict.fromkeys(PLANNERS)
                for idx,sample in enumerate(stream):
                    if idx%16==0:
                        exclusive(uuid)
                    meta = sample['meta']
                    assert ids[idx]==meta['frame']==trace[idx]['frame']==prior[idx]['frame']
                    spatial = spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
                    original = prior[idx]
                    values,layouts = {},{}
                    for name,arm in arms.items():
                        observer = observers[name]
                        observer.reset()
                        (values[name],states[name],info),cost = measured_step(arm,spatial,states[name],meta,masks[idx])
                        assert info['full_refresh']==original['refresh']
                        layout = observer.record(states[name]['deep_age'],info['full_refresh'])
                        layouts[name] = dict(**layout,core_conv_flops=cost,refresh=info['full_refresh'],
                            fingerprint=fingerprint(values[name],states[name]) if idx%16==0 else None)
                    baseline = layouts['crop_index']
                    origin = states['crop_index']
                    assert baseline['fingerprint']==original['fingerprint']
                    assert baseline['core_conv_flops']==original['core_conv_flops']
                    assert int((origin['deep_age']==0).sum())==original['written_sites']
                    for name,layout in layouts.items():
                        delta_sites = layout['work_sites']-baseline['work_sites']
                        assert layout['core_conv_flops']==baseline['core_conv_flops']+delta_sites*flops_per_site
                        assert torch.equal(origin['deep_age'],states[name]['deep_age'])
                        pairs = [(origin[k],states[name][k]) for k in (*source.state_keys,'deep_projection')]
                        pairs.extend((t,values[name][k]) for k,t in values['crop_index'].items())
                        assert all(torch.allclose(x,y,**PROJECTION_TOLERANCE) for x,y in pairs)
                        layout.update(bitwise=all(torch.equal(x,y) for x,y in pairs),
                                      maximum_abs=max(float((x-y).abs().max()) for x,y in pairs),
                                      changed=layout['jobs']!=baseline['jobs'])
                    row = dict(frame=ids[idx],refresh=original['refresh'],written_sites=original['written_sites'],
                               invalid_sites=original['invalid_sites'],arms=layouts)
                    records.append(row)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
    finally:
        for hook in hooks:
            hook.remove()
    assert len(records)==len(trace)==128
    return records,flops_per_site


def summarize(rows,geometry,parameters):
    present = [row for row in rows if row['included']]
    ordinary = [r for r in present if geometry[r['index']]['refresh'] is None]
    groups = {'all':present,'ordinary':ordinary,
              'boundary':[r for r in ordinary if geometry[r['index']]['invalid_sites']>0],
              'interior':[r for r in ordinary if geometry[r['index']]['invalid_sites']==0]}
    for name in tuple(PLANNERS)[1:]:
        groups[name+'_changed'] = [r for r in ordinary if geometry[r['index']]['arms'][name]['changed']]
    metrics = {}
    for key,group in groups.items():
        if not group:
            continue
        metrics[key] = {}
        for name in MODES:
            samples = [r['arms'][name]['model_step_ms'] for r in group]
            metrics[key][name] = dict(parameters=parameters,ap50=None,ap70=None,samples=len(samples),
                mean_ms=mean(samples),p95_ms=float(np.percentile(samples,95)),
                core_conv_flops_mean=mean(r['arms'][name]['core_conv_flops'] for r in group),
                pass_means=[mean(r['arms'][name]['model_step_ms'] for r in group if r['repeat']==p)
                            for p in range(PASSES)])
    return metrics


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    plan = checked_plan(args.timing_plan)
    assert plan['split']=='val' and len(plan['clips'])==1 and len(plan['clips'][0])==128
    phase = Path(args.prior_phase_run)
    files = {n:phase/n for n in ('trace.jsonl','contract.json','report.json','phase_index_equivalence.jsonl')}
    hashes = {n:digest(p) for n,p in files.items()}
    incumbent = json.loads(files['contract.json'].read_text())
    assert hashes['trace.jsonl']==digest(Path(args.trace))==incumbent['reference_trace_sha256']
    assert incumbent['source_sha256']==SOURCE_SHA and incumbent['plan_sha256']==digest(Path(args.timing_plan))
    assert json.loads(files['report.json'].read_text())['phase_index_equivalence_frames']==128
    prior = [json.loads(line) for line in files['phase_index_equivalence.jsonl'].read_text().splitlines()]
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    model_hash = model_digest(source)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,modes=MODES,source_sha256=SOURCE_SHA,
        reference_plan_sha256=digest(Path(args.timing_plan)),reference_phase_sha256=hashes,passes=PASSES,warmup=WARMUP,
        tolerances=PROJECTION_TOLERANCE,scope='Pillar through heads under fixed optional masks with no hooks. Phase cache and writeback masks fixed across local layouts. AP and energy unmeasured.',
        geometry='Bounded is R-295. Envelope replaces multi-call crop jobs by one aligned bounding crop if it costs less than a full region. Dense recomputes the full deep update for multi-call frames and writes only the original mask. Interior single-call path stays on the baseline.'))
    baseline = DeepBlockCache(source,TILE,'light')
    masks,trace = fixed_trace(baseline,cfg,rows,plan,out)
    assert digest(out/'trace.jsonl')==hashes['trace.jsonl']
    geometry,flops_per_site = equivalence(source,cfg,rows,plan,masks,trace,prior,out,args.uuid)
    lookup = {name:[r['arms'][name] for r in geometry] for name in PLANNERS}
    rows,checks = replay(source,cfg,rows,plan,masks,trace,out,args.uuid,MODES,False,lookup,PLANNERS)
    assert model_digest(source)==model_hash and all(digest(files[n])==v for n,v in hashes.items())
    write_json(out/'report.json',dict(status='COMPLETED',verified_scans=len(geometry),exclusive_checks=checks,
        bitwise_frames={n:sum(a['bitwise'] for a in ledger) for n,ledger in lookup.items()},
        changed_frames={n:sum(a['changed'] for a in ledger) for n,ledger in lookup.items()},
        core_flops_per_stage_site=flops_per_site,
        metrics=summarize(rows,geometry,sum(p.numel() for p in source.parameters()))))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(geometry)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','timing-plan','trace','prior-phase-run','uuid','output'):
        p.add_argument('--'+name,required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False))
        raise


if __name__ == '__main__':
    main()

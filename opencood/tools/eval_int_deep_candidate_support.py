"""Compare equal-work and bounded-support residual policies on complete zero-edge segments."""
import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch

from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_full_online import (
    AGE, DENSE, PERIOD, SCANS, SEED, check_files, commit_segment, merge_ap, native_inputs,
    read, row_identity, segment_ranges, write,
)
from opencood.tools.eval_int_deep_block_membrane import score_arm
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy
from opencood.tools.eval_int_deep_block_rank import CAP, TILE
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, compact_group_bucket, spatial_input
from opencood.tools.profile_int_deep_composed_online import build_models
from opencood.tools.profile_int_deep_phase_edges import ObservedPlanner, dense_plan
from opencood.tools.run_int_ego import loader, to_device
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.utils import eval_utils


EXPANDED = 'bounded_support'
MODES = (DENSE, EXPANDED)


def flat_ap():
    return {t:dict(tp=[],fp=[],score=[],gt=0) for t in ('0.5','0.7')}


def compile_bounded(costs, budget, cache, frame, mask, cap):
    assert cap==CAP and cache.tile==TILE and cache.stage.radius==5
    assert mask.shape[-2:]==(25,63) and frame.coarse.shape[-2:]==(50,126)
    assert not bool(mask.any()), 'The frozen zero-edge cohort has no mandatory updates'
    return 0,budget,list(costs)


def candidate_costs(path):
    geometries = [json.loads(line) for line in path.read_text().splitlines()]
    row, = [r for r in geometries if not r['mandatory']]
    base,budget,original = row['chosen_bucket']
    assert base==0 and budget==234 and len(original)==8
    costs = {tuple(map(tuple,r['pair'])):r['work_sites'] for r in row['audited'][1:] if r['work_sites']<=budget}
    assert len(costs)==36 and len({tile for pair in costs for tile in pair})==32
    assert {tuple(map(tuple,p)) for p in original}=={p for p,w in costs.items() if w==budget}
    return budget,costs


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    reference = Path(args.reference)
    plan_path,geometry = Path(args.plan),Path(args.geometry)
    files = dict(plan=plan_path,geometry=geometry,frames=reference/'frames.jsonl',prior=reference/'contract.json')
    hashes = {name:digest(path) for name,path in files.items()}
    assert hashes['frames']==args.frames_sha256 and hashes['geometry']==args.geometry_sha256
    identity = read(files['prior'])['identity']
    assert identity['source_sha256']==SOURCE_SHA and identity['plan_sha256']==hashes['plan']
    assert identity['precision']==dict(matmul_tf32=False,cudnn_tf32=False)
    budget,costs = candidate_costs(geometry)
    compiler = partial(compile_bounded,costs,budget)
    plan = read(plan_path)
    native,provenance = native_inputs(args.root)
    assert provenance['manifest_sha256']==plan['manifest_sha256'] and provenance['config_sha256']==plan['config_sha256']
    assert [row_identity(r) for r in native]==plan['ordered_stream']
    gold = [json.loads(line) for line in files['frames'].read_text().splitlines()]
    assert len(gold)==SCANS and all(g['frame']==n['frame'] and g['segment']==n['segment'] for g,n in zip(gold,native))
    cohort = [(i,lo,hi) for i,(lo,hi) in enumerate(segment_ranges(native)) if not any(r['invalid'] for r in gold[lo:hi])]
    assert len(cohort)==154 and sum(hi-lo for _,lo,hi in cohort)==3500
    if args.limit_segments is not None:
        assert 0<args.limit_segments<=len(cohort)
        cohort = cohort[:args.limit_segments]
    paths = {i:reference/'segments'/('%04d.json'%i) for i,_,_ in cohort}
    shard_hashes = {str(i):digest(p) for i,p in paths.items()}
    contract = dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                    uuid=args.uuid,gpu=torch.cuda.get_device_name(),torch=torch.__version__,source_sha256=SOURCE_SHA,
                    input_sha256=hashes,reference_segment_sha256=shard_hashes,modes=list(MODES),precision=identity['precision'],
                    cohort=cohort,stage_sites_cap=budget,pairs=len(costs),
                    scope='Frozen validation screening; both arms use the same residual scorer, indexed projection, '
                          'refresh schedule and dispatch. Only the empty-mask feasible set changes. AP and core Conv/ConvTranspose.')
    write(out/'contract.json',contract)
    check_files(plan,out)
    args.kind = 'geometry'
    source,cfg,rows,origin = build(args)
    source.period,source.max_age = PERIOD,AGE
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert origin['split']=='val' and [row_identity(r) for r in rows]==plan['ordered_stream']
    frozen = model_digest(source)
    selected = [r for _,lo,hi in cohort for r in rows[lo:hi]]
    dataset,dataloader = loader(selected,cfg,args.workers,SEED)
    stream = iter(dataloader)
    planners = {name:ObservedPlanner(dense_plan) for name in MODES}
    models = build_models(source,planners)
    hooks = [models[n].stage.register_forward_pre_hook(p.observe) for n,p in planners.items()]
    totals = {name:flat_ap() for name in ('p5',*MODES)}
    records = []
    directory = out/'segments'
    directory.mkdir()
    try:
        for i,lo,hi in cohort:
            exclusive(args.uuid)
            saved = read(paths[i])['payload']
            assert saved['records']==gold[lo:hi]
            states = dict.fromkeys(MODES,None if lo==0 else dict(scene=rows[lo-1]['segment']))
            policies = {name:ObservedPolicy(models[name],partial(CachedResidualPolicy,compiler=builder))
                        for name,builder in ((DENSE,compact_group_bucket),(EXPANDED,compiler))}
            ap = {name:flat_ap() for name in MODES}
            live = {n:{float(t):r for t,r in values.items()} for n,values in ap.items()}
            segment_records = []
            for j in range(lo,hi):
                item = next(stream)
                meta = item['meta']
                assert row_identity(meta)==plan['ordered_stream'][j]
                ego = to_device(item['ego'],torch.device('cuda'))
                spatial = spatial_input(source,ego)
                expected = gold[j]
                arms,predictions = {},{}
                for name in MODES:
                    p = planners[name]
                    p.reset()
                    policy = policies[name]
                    policy.observation = None
                    (predictions[name],states[name],info),cost = measured_step(models[name],spatial,states[name],meta,policy)
                    assert info['full_refresh']==expected['refresh']
                    layout = p.record(states[name]['deep_age'],info['full_refresh'])
                    pair = None
                    if info['full_refresh'] is None:
                        valid,required,optional = policy.observation
                        assert bool(valid.all()) and not bool(required.any())
                        assert optional is not None and torch.equal(states[name]['deep_age']==0,optional)
                        pair = policy.policy.record['chosen']
                        assert 0<layout['work_sites']==costs[pair]<=budget
                        if name==DENSE:
                            assert json.loads(json.dumps(policy.policy.record))==expected['arms'][DENSE]['selection']
                            assert optional.flatten().nonzero().flatten().tolist()==expected['written_indices']
                    if name==DENSE:
                        assert cost==expected['arms'][DENSE]['core_conv_flops']
                        assert json.loads(json.dumps(layout))==expected['arms'][DENSE]['layout']
                    arms[name] = dict(core_conv_flops=cost,work_sites=layout['work_sites'],selected=pair,
                                      written=int((states[name]['deep_age']==0).sum()))
                if meta['supervised']:
                    batch = dict(ego=ego)
                    gt = dataset.post.generate_gt_bbx(batch)
                    for name in MODES:
                        score_arm(dataset.post,batch,predictions[name],gt,live[name])
                segment_records.append(dict(index=j,frame=meta['frame'],scene=meta['scene'],segment=meta['segment'],
                                            refresh=expected['refresh'],supervised=meta['supervised'],arms=arms))
            assert ap[DENSE]==saved['ap'][DENSE],('R-309 raw detections differ',i)
            ap['p5'] = saved['ap']['p5']
            merged = dict(index=i,bounds=[lo,hi],ap=ap,frames=segment_records)
            commit_segment(directory/('%04d.json'%i),merged)
            records.extend(segment_records)
            merge_ap(totals,ap)
            write(out/'status.json',dict(status='RUNNING',phase='screen',scans=len(records),total=len(selected),segment=i))
    finally:
        for hook in hooks:
            hook.remove()
    assert next(stream,None) is None and len(records)==len(selected) and model_digest(source)==frozen
    assert all(digest(p)==hashes[name] for name,p in files.items())
    assert all(digest(p)==shard_hashes[str(i)] for i,p in paths.items())
    assert all(r['gt']==totals['p5'][t]['gt'] for a in totals.values() for t,r in a.items())
    write(out/'ap_records.json',totals)
    write(out/'sort_orders.json',{name:{t:np.argsort(-np.asarray(r['score'])).tolist()
                                      for t,r in raw.items()} for name,raw in totals.items()})
    metrics = {}
    for name,raw in totals.items():
        costs_for_arm = [gold[r['index']]['arms']['p5']['core_conv_flops'] if name=='p5' else r['arms'][name]['core_conv_flops']
                         for r in records]
        metrics[name] = dict(ap50=eval_utils.calculate_ap(raw,'0.5')[0],ap70=eval_utils.calculate_ap(raw,'0.7')[0],
                             core_conv_flops_mean=mean(costs_for_arm),parameters=sum(p.numel() for p in source.parameters()),
                             model_step_ms_mean=None)
    ordinary = [r for r in records if r['refresh'] is None]
    chosen = {name:Counter(r['arms'][name]['selected'] for r in ordinary) for name in MODES}
    write(out/'report.json',dict(status='COMPLETED',scans=len(records),labelled=sum(r['supervised'] for r in records),
                                segments=len(cohort),ordinary=len(ordinary),metrics=metrics,baseline_raw_ap_exact=True,
                                quality_control='Re-evaluated equal-work arm matches original R-309 raw detections; P5 read from old receipt.',
                                optional_pair_histogram={n:[[p,k] for p,k in c.most_common()] for n,c in chosen.items()},
                                choice_changes=sum(r['arms'][DENSE]['selected']!=r['arms'][EXPANDED]['selected'] for r in ordinary)))
    write(out/'status.json',dict(status='COMPLETED',scans=len(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','reference','geometry','frames-sha256','geometry-sha256','uuid','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit-segments',type=int)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()

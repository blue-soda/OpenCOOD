"""Audit residual, age, and offline short-window choices at matched write area."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams, tile_features
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build
from opencood.tools.probe_int_deep_block_cross_refresh import evaluate_group, periodic_offset
from opencood.tools.probe_int_deep_block_window import PROBES_PER_CLIP, WINDOW, fixed_history
from opencood.tools.run_int_ego import write_json
from opencood.tools.time_int_periodic_compact import exclusive


WORK_SITES = (234, 210)
WRITE_CELLS = 128


def fixed_groups(path):
    empty, = [r for r in map(json.loads,path.read_text().splitlines()) if not r['mandatory']]
    groups = {n:[] for n in WORK_SITES}
    for row in empty['audited'][1:]:
        pair = tuple(map(tuple,row['pair']))
        sites = row['work_sites']
        cells = sum(min(8,25-8*y)*min(8,63-8*x) for y,x in pair)
        if sites in groups and cells==WRITE_CELLS:
            groups[sites].append(pair)
    assert [len(g) for g in groups.values()]==[7,5]
    return groups


def compare_clip(cache, trace, criterion, groups, clip, scene):
    matched = 0
    for i in range(len(trace)-WINDOW+1):
        window = trace[i:i+WINDOW]
        offset = periodic_offset(window)
        if offset is None:
            continue
        root = window[0]
        frame = cache.prepare(root.spatial,root.prior,root.meta)
        required = cache.mandatory(frame).cpu()
        if bool(required.any()):
            continue
        assert required.shape[-2:]==(25,63) and frame.coarse.shape[-2:]==(50,126)
        features = tile_features(frame)
        arms = {}
        for sites,pairs in groups.items():
            scores = [sum(features[t][0] for t in pair) for pair in pairs]
            ages = [sum(features[t][4] for t in pair) for pair in pairs]
            ranks = range(len(pairs))
            residual = max(ranks,key=scores.__getitem__)
            outcomes = [evaluate_group(cache,window,pair,required,sites,float(score),criterion,offset)[0]
                        for pair,score in zip(pairs,scores)]
            for age,outcome in zip(ages,outcomes):
                outcome['age_score'] = float(age)
            exact_cost = len({tuple(r['core_flops_by_scan']) for r in outcomes})==1
            same_future_writes = all(r['future_write_mask_matches']==[True]*(WINDOW-2) for r in outcomes)
            choices = dict(residual=residual,age=max(ranks,key=ages.__getitem__),
                           current_oracle=min(ranks,key=lambda n:outcomes[n]['current_loss']),
                           window_oracle=min(ranks,key=lambda n:outcomes[n]['window_loss']))
            arms[str(sites)] = dict(groups=outcomes,selected=choices,exact_cost=exact_cost,same_future_writes=same_future_writes)
        ok = all(a['exact_cost'] and a['same_future_writes'] for a in arms.values())
        yield dict(clip=clip,scene=scene,frames=[s.meta['frame'] for s in window],refresh_at=offset,matched=ok,bins=arms)
        matched += int(ok)
        if matched==PROBES_PER_CLIP:
            break


def summary(rows):
    report = {}
    for sites in WORK_SITES:
        selected = [r['bins'][str(sites)] for r in rows]
        values = {}
        for policy in ('residual','age','current_oracle','window_oracle'):
            choices = [r['groups'][r['selected'][policy]] for r in selected]
            values[policy] = {k:mean(r[k] for r in choices) for k in
                              ('current_loss','window_loss','pre_refresh_loss','post_refresh_loss')}
        report[str(sites)] = values
    return report


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    plan_path,geometry = Path(args.plan),Path(args.geometry)
    hashes = dict(plan=digest(plan_path),geometry=digest(geometry))
    assert hashes==dict(plan=args.plan_sha256,geometry=args.geometry_sha256)
    groups = fixed_groups(geometry)
    plan = checked_plan(args.plan)
    assert plan['split']=='train' and len(plan['clips'])==12
    args.kind = 'quality'
    source,cfg,rows,origin = build(args)
    source.period,source.max_age = 5,.55
    assert origin['split']=='train'
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    cache = DeepBlockCache(source,TILE,'light',deep_projector=DeepPhaseProjection(source.core.deblocks[2],indexed=True))
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    frozen = model_digest(source)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        sources=SOURCE_SHA,inputs_sha256=hashes,gpu=torch.cuda.get_device_name(),uuid=args.uuid,
        candidates=groups,write_cells=WRITE_CELLS,window=WINDOW,windows_per_clip=PROBES_PER_CLIP,
        scope='Training-side fixed history and subsequent writes; online score versus two offline label oracles within each actual cost bin. '
              'No training, AP, time, or energy.'))
    trials,matched = 0,[]
    with (out/'trace.jsonl').open('x',buffering=1) as trace_log, (out/'windows.jsonl').open('x',buffering=1) as raw:
        for clip,ids,_,stream in clip_streams(plan,rows,cfg):
            exclusive(args.uuid)
            trace = fixed_history(cache,ids,stream,trace_log,clip)
            scene,segment = plan['scenes'][clip]
            assert all(s.meta['segment']==segment for s in trace)
            for item in compare_clip(cache,trace,criterion,groups,clip,scene):
                raw.write(json.dumps(item,allow_nan=False)+'\n')
                trials += 1
                if item['matched']:
                    matched.append(item)
            write_json(out/'status.json',dict(status='RUNNING',clips=clip+1,matched=len(matched)))
    assert model_digest(source)==frozen and hashes==dict(plan=digest(plan_path),geometry=digest(geometry))
    write_json(out/'report.json',dict(status='COMPLETED',trials=trials,matched=len(matched),
        scenes=len({r['scene'] for r in matched}),scans=sum(map(len,plan['clips'])),
        metrics=summary(matched) if matched else {},parameters=sum(p.numel() for p in source.parameters()),
        ap50=None,ap70=None,model_step_ms_mean=None))
    write_json(out/'status.json',dict(status='COMPLETED',matched=len(matched)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','plan-sha256','geometry','geometry-sha256','uuid','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as error:
        write_json(out/'status.json',dict(status='FAILED',error=repr(error)))
        raise


if __name__=='__main__':
    main()

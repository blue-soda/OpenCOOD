"""Screen fixed-budget event requests using R-288 cross-refresh outcomes."""
import argparse
from collections import Counter, defaultdict
from io import StringIO
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np

from opencood.tools.launch_int_memory import digest


TAU = .20
QUANTILE = .75
WINDOW = 3
TILE = 8
CAP = .50
EPSILON = 1e-6
RESET = ('initial', 'explicit', 'scene', 'gap')
EVENT_MODES = ('event_hold', 'event_clear')
POLICIES = ('residual', 'continuous', *EVENT_MODES, 'window_oracle')
LOSSES = ('pre_refresh_loss', 'post_refresh_loss', 'window_loss')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_inputs(args):
    root = Path(args.probe_run)
    paths = {k:root/k for k in ('trace.jsonl','frames.jsonl','contract.json','report.json')}
    paths['plan'] = Path(args.plan)
    hashes = {k:digest(p) for k,p in paths.items()}
    contract = json.loads(paths['contract.json'].read_text())
    report = json.loads(paths['report.json'].read_text())
    plan = json.loads(paths['plan'].read_text())
    assert contract['plan_sha256']==hashes['plan'] and report['status']=='COMPLETED'
    assert plan['split']=='train' and len({s[0] for s in plan['scenes']})==len(plan['clips'])==12
    assert (contract['tile'],contract['cap'],contract['period'],contract['max_age'],contract['window']) == (TILE,CAP,5,.55,WINDOW)
    source = jsonl(paths['trace.jsonl'])
    assert [(r['clip'],r['frame']) for r in source]==[(i,f) for i,clip in enumerate(plan['clips']) for f in clip]
    trace = defaultdict(list)
    for row in source:
        trace[row['clip']].append(row)
    all_windows = jsonl(paths['frames.jsonl'])
    assert len(all_windows)==report['windows']
    windows = [w for w in all_windows if w['matched']]
    assert len(windows)==report['matched_windows']
    assert len({w['frames'][0] for w in windows})==len(windows)
    for row in windows:
        clip = row['clip']
        idx = plan['clips'][clip].index(row['frames'][0])
        fixed = trace[clip][idx:idx+WINDOW]
        reasons = [r['refresh'] for r in fixed]
        refresh = row['periodic_offset']
        assert row['frames']==[r['frame'] for r in fixed] and row['scene']==plan['scenes'][clip][0]
        assert row['refresh_reasons']==reasons and reasons.count('period')==1 and reasons[refresh]=='period'
        assert reasons[0] is None and all(x in (None,'period') for x in reasons)
        assert row['cost_match'] and row['future_writes_match']
        candidates = row['groups']
        assert len(candidates)>=2 and len({tuple(g['core_flops_by_scan']) for g in candidates})==1
        assert all(g['future_write_mask_matches']==[True] for g in candidates)
        assert len({tuple(tuple(k) for k in g['pair']) for g in candidates})==len(candidates)
        for group in candidates:
            labels = group['loss_by_scan']
            assert [v is not None for v in labels]==[r['supervised'] for r in fixed]
            assert group['current_loss']==labels[0]
            for name,block in zip(LOSSES,(labels[:refresh],labels[refresh:],labels)):
                assert abs(mean(v for v in block if v is not None)-group[name])<1e-12
        rank = range(len(candidates))
        assert row['selected']==dict(residual=max(rank,key=lambda i:candidates[i]['residual_score']),
            current_oracle=min(rank,key=lambda i:candidates[i]['current_loss']),
            window_oracle=min(rank,key=lambda i:candidates[i]['window_loss']))
    for mode in ('residual','window_oracle'):
        for field in LOSSES:
            value = mean(w['groups'][w['selected'][mode]][field] for w in windows)
            assert abs(value-report['mean_loss'][mode][field])<1e-12
    return paths,hashes,contract,plan,trace,windows


def warp_indices(cache,scan,torch,block_range,warp):
    prior = scan.prior
    size = prior['deep'].shape[-2:]
    bounds = block_range(cache.source.bounds,scan.spatial.shape[-2:],size,8)
    grid = torch.arange(1,1+size[0]*size[1],device=scan.spatial.device,dtype=scan.spatial.dtype).reshape(1,1,*size)
    mapped,valid = warp(grid,prior['pose'],scan.meta['pose'],bounds)
    return (mapped.flatten().to(torch.int64)-1).tolist(),valid


def capture(args,plan,trace,windows,reference,out):
    import torch

    from opencood.models.sub_modules.int_block_cache import block_lidar_range
    from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
    from opencood.models.sub_modules.int_feature_memory import warp_bev
    from opencood.tools.check_int_pyramid_cache import measured_step
    from opencood.tools.check_int_temporal_feature import model_digest
    from opencood.tools.eval_int_deep_block_rank import checked_plan, clip_streams, tile_features
    from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket, masks_by_tile
    from opencood.tools.probe_int_deep_block_window import fixed_history

    assert reference['source_sha256']==SOURCE_SHA and plan==checked_plan(args.plan)
    args.kind = 'quality'
    source,cfg,rows,provenance = build(args)
    assert provenance['split']=='train'
    source.period,source.max_age = 5,.55
    original = model_digest(source)
    model = DeepBlockCache(source,TILE,'light')
    at_frame = {w['frames'][0]:w for w in windows}
    all_observations = []
    with torch.no_grad(), (out/'observations.jsonl').open('x',buffering=1) as raw:
        for clip,ids,_,stream in clip_streams(plan,rows,cfg):
            journal = StringIO()
            neutral = fixed_history(model,ids,stream,journal,clip)
            assert [json.loads(line) for line in journal.getvalue().splitlines()]==trace[clip]
            scene,segment = plan['scenes'][clip]
            for i,scan in enumerate(neutral):
                meta = scan.meta
                assert meta['segment']==segment
                row = dict(frame=meta['frame'],clip=clip,scene=scene,refresh=scan.refresh)
                if scan.refresh not in RESET:
                    mapping,valid = warp_indices(model,scan,torch,block_lidar_range,warp_bev)
                    dt = (int(meta['timestamp_us'])-int(scan.prior['timestamp_us']))/1e6
                    assert 0 < dt <= .25
                    row.update(dt_s=dt,shape=list(valid.shape[-2:]),previous_indices=mapping,
                        valid=valid.flatten().nonzero().flatten().tolist())
                if scan.refresh is None:
                    frame = model.prepare(scan.spatial,scan.prior,meta)
                    assert torch.equal(frame.valid,valid)
                    required = model.mandatory(frame).cpu()
                    features = tile_features(frame)[...,0]
                    row.update(residual=features.tolist(),eligible=list(masks_by_tile(required,TILE)),
                        written=scan.write_mask.flatten().nonzero().flatten().tolist())
                    window = at_frame.pop(meta['frame'],None)
                    if window is not None:
                        base,sites,pairs = group_bucket(model,frame,required,CAP)
                        candidates = window['groups']
                        assert (base,sites,required.numel())==(window['base_sites'],window['sites'],window['lattice_sites'])
                        assert pairs==[tuple(map(tuple,g['pair'])) for g in candidates]
                        assert [float(features[a])+float(features[b]) for a,b in pairs]==[g['residual_score'] for g in candidates]
                        selected = candidates[window['selected']['residual']]
                        state = scan.prior
                        for j,step in enumerate(neutral[i:i+WINDOW]):
                            request = None
                            if j==0:
                                request = add_tiles(torch.zeros_like(required),pairs[window['selected']['residual']],TILE).cuda()
                            elif step.refresh is None:
                                def fixed_request(frame, mandatory):
                                    combined = mandatory if step.request is None else (mandatory | step.request)
                                    assert torch.equal(combined,step.write_mask), step.meta['frame']
                                    return step.request

                                request = fixed_request
                            (_,state,info),core = measured_step(model,step.spatial,state,step.meta,request)
                            assert core==selected['core_flops_by_scan'][j] and info['full_refresh']==step.refresh
                        row['verified_residual_cost'] = selected['core_flops_by_scan']
                all_observations.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(out/'status.json',dict(status='RUNNING',phase='causal_capture',clips=clip+1))
    assert not at_frame and model_digest(source)==original
    identity = dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
        parameters=sum(p.numel() for p in source.parameters()),source_weight_digest=original,
        observations_sha256=digest(out/'observations.jsonl'))
    return all_observations,identity


def tile_reduce(values, valid):
    height,width = values.shape
    means = np.zeros(((height+TILE-1)//TILE,(width+TILE-1)//TILE))
    for y,x in np.ndindex(means.shape):
        sl = np.s_[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE]
        support = valid[sl]
        if support.any():
            means[y,x] = values[sl][support].mean()
    return means


def lift(tiles,shape):
    return np.repeat(np.repeat(tiles,TILE,axis=0),TILE,axis=1)[:shape[0],:shape[1]]


def replay_history(observations, threshold=None, clear_at_period=False):
    evidence = pending = None
    snapshots = {}
    for row in observations:
        refresh = row['refresh']
        if refresh in RESET:
            evidence = pending = None
            continue
        shape = tuple(row['shape'])
        mapping = np.asarray(row['previous_indices'],dtype=np.int64)
        valid = np.zeros(mapping.shape,dtype=bool)
        valid[row['valid']] = True
        covered = valid & (mapping>=0)
        memory = np.zeros(mapping.shape)
        requests = np.zeros(mapping.shape,dtype=bool)
        if evidence is not None:
            memory[covered] = evidence.ravel()[mapping[covered]]*math.exp(-row['dt_s']/TAU)
            requests[covered] = pending.ravel()[mapping[covered]]
        evidence,pending = memory.reshape(shape),requests.reshape(shape)
        if refresh is not None:
            assert refresh in ('period','age')
            pending.fill(False)
            if clear_at_period:
                evidence.fill(0.)
            continue
        valid = valid.reshape(shape)
        evidence += lift(np.asarray(row['residual']),shape)*valid
        score = tile_reduce(evidence,valid)
        carried = tile_reduce(pending,valid)>=.5
        crossed = score>=threshold if threshold is not None else np.zeros_like(carried)
        eligible = [tuple(t) for t in row['eligible']]
        requested = carried | crossed
        written = np.zeros(mapping.shape,dtype=bool)
        written[row['written']] = True
        written = written.reshape(shape)
        pending = lift(requested,shape) & valid
        clear = int((pending & written).sum())
        pending[written] = False
        evidence[written] = 0.
        counters = dict(requested=sum(bool(requested[k]) for k in eligible),
            latched_without_recross=sum(bool(carried[k] and not crossed[k]) for k in eligible),
            pending_after_feedback=sum(bool(v) for v in (tile_reduce(pending,valid)>=.5).flatten()),
            pending_cells_cleared=clear)
        snapshots[row['frame']] = (score,requested,counters)
    return snapshots


def compare(values):
    return dict(mean=mean(values),wins=sum(v>EPSILON for v in values),ties=sum(abs(v)<=EPSILON for v in values),
                losses=sum(v < -EPSILON for v in values))


def score_windows(observations, windows, out):
    by_scene = defaultdict(list)
    for row in observations:
        by_scene[row['scene']].append(row)
    matched = defaultdict(list)
    for window in windows:
        matched[window['scene']].append(window)
    base = {scene:replay_history(rows) for scene,rows in by_scene.items()}
    values = {scene:[base[scene][row['frame']][0][tuple(k)] for row in rows if row['refresh'] is None
                     for k in row['eligible']] for scene,rows in by_scene.items()}
    selections,folds = [],[]
    with (out/'predictions.jsonl').open('x',buffering=1) as raw:
        for scene in sorted(matched):
            training = sorted(set(by_scene)-{scene})
            training_samples = [v for other in training for v in values[other]]
            theta = max(EPSILON,float(np.quantile(training_samples,QUANTILE)))
            hold = replay_history(by_scene[scene],theta)
            clear = replay_history(by_scene[scene],theta,clear_at_period=True)
            folds.append(dict(held_scene=scene,threshold=theta,training_scenes=training,threshold_samples=len(training_samples),
                hold_request_totals={k:sum(v[2][k] for v in hold.values()) for k in next(iter(hold.values()))[2]},
                clear_request_totals={k:sum(v[2][k] for v in clear.values()) for k in next(iter(clear.values()))[2]}))
            for window in matched[scene]:
                frame = window['frames'][0]
                candidates = window['groups']
                pairs = [tuple(map(tuple,g['pair'])) for g in candidates]
                oracle = window['selected']['window_oracle']
                scores = {'residual':[(0,g['residual_score']) for g in candidates]}
                controllers = {}
                for mode,replayed in zip(POLICIES[1:4],(base[scene],hold,clear)):
                    energy,pending,counters = replayed[frame]
                    scores[mode] = [(sum(int(pending[k]) for k in pair),sum(float(energy[k]) for k in pair)) for pair in pairs]
                    controllers[mode] = counters
                selection = {}
                order = range(len(pairs))
                for mode,score in scores.items():
                    ranked = sorted(order,key=score.__getitem__,reverse=True)
                    index = ranked[0]
                    selection[mode] = dict(index=index,pair=pairs[index],oracle_rank=1+ranked.index(oracle),
                        scores=score,requested_tiles=score[index][0],loss={k:candidates[index][k] for k in LOSSES})
                assert selection['residual']['index']==window['selected']['residual']
                selection['window_oracle'] = dict(index=oracle,pair=pairs[oracle],oracle_rank=1,
                    loss={k:candidates[oracle][k] for k in LOSSES})
                row = dict(scene=scene,frames=window['frames'],periodic_offset=window['periodic_offset'],
                    core_flops_by_scan=candidates[oracle]['core_flops_by_scan'],selection=selection,controllers=controllers)
                selections.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
    return selections,folds


def summarize(rows):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row['scene']].append(row)
    summary = {m:dict(means={f:mean(r['selection'][m]['loss'][f] for r in rows) for f in LOSSES},
        oracle_top1=sum(r['selection'][m]['oracle_rank']==1 for r in rows),
        oracle_top3=sum(r['selection'][m]['oracle_rank']<=3 for r in rows),
        oracle_mean_rank=mean(r['selection'][m]['oracle_rank'] for r in rows)) for m in POLICIES}
    comparisons = {}
    for left,right in (('residual','continuous'),('residual','event_hold'),('residual','event_clear'),
                       ('continuous','event_hold'),('event_clear','event_hold'),('residual','window_oracle')):
        result = dict(changed=sum(r['selection'][left]['index']!=r['selection'][right]['index'] for r in rows))
        for field in LOSSES:
            values = {scene:[r['selection'][left]['loss'][field]-r['selection'][right]['loss'][field] for r in records]
                      for scene,records in by_scene.items()}
            per_scene = {scene:mean(v) for scene,v in values.items()}
            result[field] = dict(per_window=compare([v for group in values.values() for v in group]),
                                 per_scene=compare(list(per_scene.values())),scene_means=per_scene)
        comparisons[right+'_over_'+left] = result
    return dict(summary=summary,comparisons=comparisons)


def run(args,out):
    paths,hashes,reference,plan,trace,windows = load_inputs(args)
    identity = dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        probe_commit=reference['code_commit'],source_sha256=reference['source_sha256'],input_sha256=hashes,
        scope='One causal selection from a frozen earlier trajectory; candidate losses are R-288 off-policy outcomes. No repeated policy interventions, training, AP, or spent model latency.',
        tau_s=TAU,threshold_quantile=QUANTILE,threshold_calibration='Target-free evidence on ordinary eligible tiles from all nonheld scenes only.',
        feedback='Ordinary executed writes clear evidence and latched requests on those cells. Period/age clears pending requests everywhere, with held or cleared residual evidence. Scene/reset/gap always clears both.',
        score='Leaky continuous ranks summed valid tile evidence. Event variants prioritize count of requested pair tiles, then identical leaky evidence; a latched tile persists with at least half its valid cells after nearest alignment.')
    write_json(out/'contract.json',identity)
    if args.inspect_only:
        info = dict(status='COMPLETED',inspect_only=True,matched_windows=len(windows),
            scenes=len({r['scene'] for r in windows}),candidate_groups=sum(len(r['groups']) for r in windows),
            periodic_offsets=dict(Counter(r['periodic_offset'] for r in windows)),
            residual_and_oracle={m:{k:mean(r['groups'][r['selected'][m]][k] for r in windows) for k in LOSSES}
                                 for m in ('residual','window_oracle')})
    else:
        observed,device = capture(args,plan,trace,windows,reference,out)
        rows,folds = score_windows(observed,windows,out)
        info = dict(status='COMPLETED',inspect_only=False,matched_windows=len(rows),scenes=len(folds),folds=folds,
            equal_core_flops_all_windows=True,baseline_archived_residual_exact=True,
            mean_core_flops_by_scan=[mean(r['core_flops_by_scan'][i] for r in rows) for i in range(WINDOW)],
            ap50=None,ap70=None,model_step_ms_mean=None,**device,**summarize(rows))
    assert all(digest(paths[k])==v for k,v in hashes.items())
    write_json(out/'report.json',info)
    write_json(out/'status.json',dict(status='COMPLETED',matched_windows=len(windows),inspect_only=args.inspect_only))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','probe-run','output'):
        parser.add_argument('--'+name,required=True)
    for name in ('root','checkpoint','trained','readout'):
        parser.add_argument('--'+name)
    parser.add_argument('--inspect-only',action='store_true')
    args = parser.parse_args()
    if not args.inspect_only and any(getattr(args,k) is None for k in ('root','checkpoint','trained','readout')):
        parser.error('Policy evaluation requires the frozen source weight locations')
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

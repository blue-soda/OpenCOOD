"""Lock a metadata-selected validation cohort and audit online indexed deep reuse."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np

from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest


CLIPS = 12
LENGTH = 48
WARMUP = 8
EDGE_CLIPS = 8
PERIOD, MAX_AGE, TILE = 5, .55, 8
MODES = ('p5','indexed_crop_online','indexed_dense_online')
THRESHOLDS = (.5,.7)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def planar(value):
    pose = np.asarray(value)
    angle = math.atan2(pose[1,0],pose[0,0])
    c,s = math.cos(angle),math.sin(angle)
    return np.array(((c,-s,pose[0,3]),(s,c,pose[1,3]),(0.,0.,1.)))


def lattice(config):
    bounds = config['model']['args']['lidar_range']
    vx,vy = config['model']['args']['voxel_size'][:2]
    xmin,ymin,_,xmax,ymax,_ = bounds
    width,height = round((xmax-xmin)/vx),round((ymax-ymin)/vy)
    dx,dy = (xmax-xmin)/width,(ymax-ymin)/height
    nx,ny = (width+7)//8,(height+7)//8
    x0,y0 = xmin-3.5*dx,ymin-3.5*dy
    extent = nx*8*dx,ny*8*dy
    xx,yy = np.meshgrid(x0+(np.arange(nx,dtype=np.float32)+.5)*(8*dx),
                        y0+(np.arange(ny,dtype=np.float32)+.5)*(8*dy),indexing='xy')
    points = np.stack((xx.ravel(),yy.ravel(),np.ones(xx.size,dtype=np.float32)),1)
    return points,np.array((x0,y0),dtype=np.float32),np.asarray(extent,dtype=np.float32)


def invalid_counts(rows, grid):
    points,origin,extent = grid
    result = [0]
    old = planar(rows[0]['pose'])
    for row in rows[1:]:
        current = planar(row['pose'])
        transfer = np.linalg.solve(old,current).astype(np.float32)
        mapped = (points @ transfer.T)[:,:2]
        normalized = 2*(mapped-origin)/extent-1
        result.append(int(np.any(np.abs(normalized)>1,axis=1).sum()))
        old = current
    return result


def exposure(clip, invalid):
    edge = cells = 0
    latest = clip[0]['timestamp_us']
    since = 0
    for i,row in enumerate(clip[1:],start=1):
        timestamp = row['timestamp_us']
        reset = row['reset_before'] or not 0 < (timestamp-clip[i-1]['timestamp_us'])/1e6 <= .25
        if reset or since+1 >= PERIOD or (timestamp-latest)/1e6 >= MAX_AGE-1e-6:
            latest,since = timestamp,0
        else:
            since += 1
            if i>=WARMUP:
                edge += int(invalid[i]>0)
                cells += invalid[i]
    return edge,cells


def choose_clips(rows, config, exclude):
    by_id = {row['frame']:row for row in rows}
    blocked = {by_id[f]['segment'] for plan in exclude for clip in plan['clips'] for f in clip}
    seen = {by_id[f]['scene'] for plan in exclude for clip in plan['clips'] for f in clip}
    segments = defaultdict(list)
    for row in rows:
        if row['segment'] not in blocked:
            segments[row['segment']].append(row)
    grid = lattice(config)
    candidates = []
    for segment,frames in sorted(segments.items()):
        if len(frames)<LENGTH:
            continue
        counts = invalid_counts(frames,grid)
        starts = sorted(set(range(0,len(frames)-LENGTH+1,LENGTH)) | {len(frames)-LENGTH})
        for start in starts:
            data = frames[start:start+LENGTH]
            edge,cells = exposure(data,counts[start:start+LENGTH])
            scene = data[0]['scene']
            candidates.append(dict(scene=scene,segment=segment,frames=data,previous_scene=scene in seen,
                                   predicted_edge_ordinary=edge,predicted_invalid_cells=cells))
    available = len({r['scene'] for r in candidates})
    if available < CLIPS:
        raise ValueError('Only %d eligible unique scenes after excluding prior segments; need %d' % (available,CLIPS))
    picked = []
    scenes = set()
    keys = lambda row:(row['scene'],row['segment'],row['frames'][0]['frame'])
    edge = sorted((x for x in candidates if x['predicted_edge_ordinary']>0),
        key=lambda x:(x['previous_scene'],-x['predicted_edge_ordinary'],-x['predicted_invalid_cells'],*keys(x)))
    for item in edge:
        if item['scene'] not in scenes:
            picked.append(dict(item,stratum='edge'))
            scenes.add(item['scene'])
        if len(picked)==EDGE_CLIPS:
            break
    controls = sorted(candidates,key=lambda x:(x['predicted_edge_ordinary']>0,x['previous_scene'],
                          x['predicted_edge_ordinary'],x['predicted_invalid_cells'],*keys(x)))
    for item in controls:
        if item['scene'] not in scenes:
            picked.append(dict(item,stratum='low'))
            scenes.add(item['scene'])
        if len(picked)==CLIPS:
            break
    assert len(picked)==CLIPS
    if not any(x['stratum']=='edge' for x in picked):
        raise ValueError('Validation candidate segments contain no metadata-predicted ordinary edge exposure')
    return picked,dict(excluded_segments=sorted(blocked),older_scene_ids=sorted(seen),eligible_scenes=available,
                       candidates=len(candidates),eligible_edge_scenes=len({r['scene'] for r in edge}))


def prepare(args,out):
    import yaml

    root = Path(args.root)/'full_gru'
    sources = read(root/'manifest.json')['contract']
    runtime = read(root/'effective_runtime.json')['arguments']
    manifest = Path(artifact_path(runtime['manifest']))
    config = Path(artifact_path(runtime['config']))
    assert digest(manifest)==sources['manifest_sha256'] and digest(config)==sources['config_sha256']
    exclusions = [read(p) for p in args.exclude_plan]
    assert all(p['split']=='val' for p in exclusions)
    selected,eligible = choose_clips(read(manifest)['val'],yaml.safe_load(config.read_text()),exclusions)
    selected_rows = [row for item in selected for row in item['frames']]
    assert len({r['frame'] for r in selected_rows})==CLIPS*LENGTH
    files = sorted({r[k] for r in selected_rows for k in ('pcd','label') if r[k] is not None})
    input_hashes = {path:digest(Path(artifact_path(path))) for path in files}
    plan = dict(split='val',clips=[[r['frame'] for r in x['frames']] for x in selected],
        scenes=[[x['scene'],x['segment']] for x in selected],sha256=input_hashes,
        selection=[{k:v for k,v in x.items() if k!='frames'} for x in selected],warmup=WARMUP,
        source_manifest_sha256=sources['manifest_sha256'],source_config_sha256=sources['config_sha256'],
        excluded_plans_sha256=[digest(Path(p)) for p in args.exclude_plan],**eligible)
    write(out/'plan.json',plan)
    labels = [sum(r['supervised'] for r in item['frames'][WARMUP:]) for item in selected]
    report = dict(status='COMPLETED',scenes=CLIPS,scans=len(selected_rows),warmup_each=WARMUP,
        eligible_scored_scans=CLIPS*(LENGTH-WARMUP),known_labelled_scans_post_warmup=sum(labels),
        known_labelled_by_scene=dict(zip((x['scene'] for x in selected),labels)),
        selected_scales=plan['selection'],plan_sha256=digest(out/'plan.json'),files_hashed=len(input_hashes),**eligible)
    write(out/'report.json',report)


def selection_equal(old,new):
    return all(old[k]==new[k] for k in ('chosen','pairs','mandatory_sites','planned_sites'))


def evaluate(args,out):
    import torch
    from torch import nn

    from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
    from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
    from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
    from opencood.tools.check_int_pyramid_cache import measured_step
    from opencood.tools.check_int_temporal_feature import model_digest
    from opencood.tools.eval_int_deep_block_membrane import score_arm
    from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy, original_online_policy
    from opencood.tools.eval_int_deep_block_rank import checked_plan, clip_streams
    from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
    from opencood.tools.profile_int_deep_block_cache import PROJECTION_TOLERANCE
    from opencood.tools.profile_int_deep_phase_edges import ObservedPlanner, dense_plan
    from opencood.tools.run_int_ego import to_device
    from opencood.tools.train_int_deep_block_adapt import pack_mask
    from opencood.utils import eval_utils

    plan = checked_plan(args.plan)
    plan_sha = digest(Path(args.plan))
    assert plan['split']=='val' and plan['warmup']==WARMUP and len(plan['clips'])==CLIPS
    assert len({s[0] for s in plan['scenes']})==CLIPS and all(len(c)==LENGTH for c in plan['clips'])
    args.kind = 'geometry'
    source,cfg,rows,provenance = build(args)
    source.period,source.max_age = PERIOD,MAX_AGE
    assert provenance['contract']['manifest_sha256']==plan['source_manifest_sha256']
    assert provenance['contract']['config_sha256']==plan['source_config_sha256']
    frozen = model_digest(source)
    crop,dense = MODES[1:]
    observers = {crop:ObservedPlanner(plan_host_coalesced_refresh),dense:ObservedPlanner(dense_plan)}
    models = {'p5':source}
    for mode,observer in observers.items():
        models[mode] = DeepBlockCache(source,TILE,'light',planner=observer,crop_first=True,
            deep_projector=DeepPhaseProjection(source.core.deblocks[2],indexed=True))
    hooks = [models[m].stage.register_forward_pre_hook(observer.observe) for m,observer in observers.items()]
    per_site = sum(2*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                   for m in models[crop].stage.modules() if isinstance(m,nn.Conv2d))
    stats = {m:{t:dict(tp=[],fp=[],score=[],gt=0) for t in THRESHOLDS} for m in MODES}
    records,slices = [],[]
    write(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        plan_sha256=plan_sha,source_sha256=SOURCE_SHA,device=torch.cuda.get_device_name(),torch=torch.__version__,
        period=PERIOD,max_age=MAX_AGE,tile=TILE,arms=MODES,warmup_each=WARMUP,planned_flops_per_site=per_site,
        parity_tolerance=PROJECTION_TOLERANCE,original_policy_checked='Every ordinary frame, same prepared tensors as cached baseline; audit only.',
        scope='Frozen metadata-enriched validation cohort. Geometry-cached online residual in both indexed layouts, distinct recurrent states, exact write masks, measured core Conv/ConvTranspose. No timing, training or energy.'))
    try:
        with torch.no_grad(), (out/'frames.jsonl').open('x',buffering=1) as raw:
            for clip,ids,dataset,stream in clip_streams(plan,rows,cfg):
                state = dict.fromkeys(MODES)
                scene,segment = plan['scenes'][clip]
                policies = {m:ObservedPolicy(models[m],CachedResidualPolicy) for m in observers}
                original = original_online_policy(models[crop])

                def baseline_policy(frame, mandatory):
                    chosen = policies[crop](frame,mandatory)
                    gold = original(frame,mandatory)
                    assert (chosen is None)==(gold is None)
                    assert chosen is None or torch.equal(chosen,gold)
                    assert selection_equal(original.record,policies[crop].policy.record)
                    return chosen

                start = {m:{t:(len(d['score']),d['gt']) for t,d in arm.items()} for m,arm in stats.items()}
                for i,sample in enumerate(stream):
                    meta = sample['meta']
                    assert ids[i]==meta['frame'] and meta['segment']==segment
                    ego = to_device(sample['ego'],torch.device('cuda'))
                    spatial = spatial_input(source,ego)
                    requests = dict(p5=None,**{crop:baseline_policy,dense:policies[dense]})
                    predictions,arm = {},{}
                    reasons = []
                    for mode,model in models.items():
                        if mode in observers:
                            observers[mode].reset()
                            policies[mode].observation = None
                        (pred,state[mode],info),core = measured_step(model,spatial,state[mode],meta,requests[mode])
                        assert all(bool(torch.isfinite(t).all()) for t in pred.values())
                        refresh = info['full_refresh']
                        predictions[mode] = pred
                        reasons.append(refresh)
                        buf = state[mode]
                        names = (*source.state_keys,'deep_age','deep_projection') if mode in observers else source.state_keys
                        layout = observers[mode].record(buf['deep_age'],refresh) if mode in observers else None
                        arm[mode] = dict(core_conv_flops=core,state_bytes=sum(buf[k].numel()*buf[k].element_size() for k in names),layout=layout)
                    assert len(set(reasons))==1
                    refresh = reasons[0]
                    invalid,written = 0,None
                    if refresh is None:
                        va,mandatory,optional = policies[crop].observation
                        vb,ab,bb = policies[dense].observation
                        assert torch.equal(va,vb) and torch.equal(mandatory,ab)
                        assert (optional is None)==(bb is None) and (bb is None or torch.equal(optional,bb))
                        assert selection_equal(policies[crop].policy.record,policies[dense].policy.record)
                        written = mandatory if optional is None else mandatory|optional
                        assert torch.equal(written,state[crop]['deep_age']==0) and torch.equal(written,state[dense]['deep_age']==0)
                        invalid = int((~va).sum())
                        for m in observers:
                            arm[m]['selection'] = policies[m].policy.record
                    else:
                        assert all(policy.observation is None for policy in policies.values())
                    base_work,new_work = (arm[m]['layout']['work_sites'] for m in (crop,dense))
                    assert arm[dense]['core_conv_flops']-arm[crop]['core_conv_flops']==(new_work-base_work)*per_site
                    fields = [(state[crop][k],state[dense][k]) for k in (*source.state_keys,'deep_age','deep_projection')]
                    fields += [(predictions[crop][k],predictions[dense][k]) for k in predictions[crop]]
                    assert all(torch.allclose(a,b,**PROJECTION_TOLERANCE) for a,b in fields)
                    exact = all(torch.equal(a,b) for a,b in fields)
                    score = i>=WARMUP and bool(meta['supervised'])
                    if score:
                        batch = dict(ego=ego)
                        gt = dataset.post.generate_gt_bbx(batch)
                        for m in MODES:
                            score_arm(dataset.post,batch,predictions[m],gt,stats[m])
                    row = dict(clip=clip,scene=scene,frame=meta['frame'],index=i,scored=score,supervised=bool(meta['supervised']),
                        refresh=refresh,invalid=invalid,written=pack_mask(written),arms=arm,layout_states_heads_bitwise=exact,
                        layout_max_abs=0. if exact else max(float((a-b).abs().max()) for a,b in fields))
                    records.append(row)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
                slices.append(dict(scene=scene,arms={m:{t:[start[m][t][0],len(d['score']),d['gt']-start[m][t][1]]
                              for t,d in arm.items()} for m,arm in stats.items()}))
                write(out/'status.json',dict(status='RUNNING',phase='quality',frames=len(records),clips=clip+1))
    finally:
        for hook in hooks:
            hook.remove()
    assert model_digest(source)==frozen and digest(Path(args.plan))==plan_sha
    assert len(records)==CLIPS*LENGTH and stats['p5'][.7]['gt']>0
    bitwise = sum(r['layout_states_heads_bitwise'] for r in records)
    score_exact = stats[crop]==stats[dense]
    if bitwise==len(records):
        assert score_exact
    raw_path = out/'ap_records.json'
    write(raw_path,stats)
    write(out/'sort_orders.json',dict(source_sha256=digest(raw_path),numpy_version=np.__version__,
        orders={m:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in a.items()} for m,a in stats.items()}))
    write(out/'scene_slices.json',slices)
    ordinary = [r for r in records if r['refresh'] is None]
    metrics = {m:dict(parameters=sum(p.numel() for p in source.parameters()),ap50=eval_utils.calculate_ap(stats[m],.5)[0],
        ap70=eval_utils.calculate_ap(stats[m],.7)[0],state_bytes_max=max(r['arms'][m]['state_bytes'] for r in records),
        model_step_ms_mean=None,core_conv_flops_mean=mean(r['arms'][m]['core_conv_flops'] for r in records),
        ordinary_core_conv_flops_mean=mean(r['arms'][m]['core_conv_flops'] for r in ordinary)) for m in MODES}
    for m in (crop,dense):
        metrics[m].update(cache_hits=sum(r['arms'][m]['selection']['geometry_cache_hit'] for r in ordinary),
            mean_stage_calls=mean(r['arms'][m]['layout']['calls'] for r in ordinary))
    write(out/'report.json',dict(status='COMPLETED',scans=len(records),scored_scans=sum(r['scored'] for r in records),
        supervised_scans=sum(r['supervised'] for r in records),gt=stats['p5'][.7]['gt'],scenes=CLIPS,warmup_each=WARMUP,
        layout_bitwise_frames=bitwise,layout_raw_ap_exact=score_exact,layout_max_abs=max(r['layout_max_abs'] for r in records),
        original_callback_ordinary_frames=len(ordinary),exact_write_masks=True,weights_unchanged=True,
        actual_edge_ordinary_scans=sum(r['invalid']>0 for r in ordinary),
        actual_edge_ordinary_post_warmup=sum(r['invalid']>0 and r['index']>=WARMUP for r in ordinary),
        actual_edge_scenes=len({r['scene'] for r in ordinary if r['invalid']>0}),
        ordinary_scans=len(ordinary),dense_changed_ordinary=sum(r['arms'][crop]['layout']['jobs']!=r['arms'][dense]['layout']['jobs'] for r in ordinary),
        refresh_reasons=dict(Counter(r['refresh'] or 'ordinary' for r in records)),metrics=metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--exclude-plan',action='append',default=[])
    for name in ('root','output'):
        parser.add_argument('--'+name,required=True)
    for name in ('plan','checkpoint','trained','readout'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if not args.plan_only and any(getattr(args,k) is None for k in ('plan','checkpoint','trained','readout')):
        parser.error('Quality mode requires a locked plan and paths for the three frozen weights')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='RUNNING',phase='metadata' if args.plan_only else 'quality'))
    try:
        if args.plan_only:
            prepare(args,out)
        else:
            evaluate(args,out)
        write(out/'status.json',dict(status='COMPLETED',plan_only=args.plan_only))
    except Exception as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()

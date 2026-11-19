"""Audit exact online residual scheduling and optionally time its full model path."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_local_state_cache import relative_change_score
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_membrane import score_arm
from opencood.tools.eval_int_deep_block_rank import CAP, TILE, CausalPolicy, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import PASSES, fingerprint
from opencood.tools.time_int_deep_phase_transfer import DEVICE, WARMUP, inputs
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.tools.train_int_deep_block_adapt import EPOCHS, local_request, pack_mask, trace_masks
from opencood.utils import eval_utils


MODES = ('p5', 'fixed_crop', 'indexed_online')
TIMING_MODES = ('p5', 'indexed_fixed', 'indexed_online')
THRESHOLDS = (.5, .7)
MAX_GEOMETRIES = 64


def read(path):
    return json.loads(Path(path).read_text())


def original_online_policy(cache):
    """The unchanged callable used by R-292 trace compilation."""
    return CausalPolicy(cache, 'residual', None)


class CachedResidualPolicy:
    """Reuse geometry buckets while scoring live residuals on every ordinary frame."""

    def __init__(self, cache):
        self.cache = cache
        self.layouts = OrderedDict()
        self.record = None
        self.hits = self.misses = 0

    def __call__(self, frame, required):
        mask = required.cpu()
        key = (tuple(frame.coarse.shape[-2:]), tuple(mask.shape[-2:]), mask.numpy().tobytes())
        hit = key in self.layouts
        if hit:
            self.hits += 1
            self.layouts.move_to_end(key)
        else:
            self.misses += 1
            self.layouts[key] = group_bucket(self.cache,frame,mask,CAP)
            if len(self.layouts)>MAX_GEOMETRIES:
                self.layouts.popitem(last=False)
        mandatory_sites,sites,pairs = self.layouts[key]
        self.record = dict(mandatory_sites=mandatory_sites,planned_sites=mandatory_sites,
                           pairs=len(pairs),chosen=None,geometry_cache_hit=hit)
        if not pairs:
            return None
        shape = frame.aligned.shape[-2:]
        current = F.adaptive_avg_pool2d(frame.coarse,shape)
        old = F.adaptive_avg_pool2d(frame.old_coarse,shape)
        residual = torch.log1p(relative_change_score(current,old,TILE))[0,0].cpu().numpy().astype(np.float64)
        pair = max(pairs,key=lambda p:residual[p[0]]+residual[p[1]])
        self.record.update(chosen=pair,planned_sites=sites)
        return add_tiles(torch.zeros_like(mask),pair,TILE).to(required.device)


class ObservedPolicy:
    def __init__(self, cache, factory):
        self.policy = factory(cache)
        self.observation = None

    def __call__(self, frame, mandatory):
        assert self.observation is None, 'Expected one in-step policy call'
        optional = self.policy(frame,mandatory)
        self.observation = (frame.valid,mandatory,optional)
        return optional


def local_model(source, indexed=False):
    projector = DeepPhaseProjection(source.core.deblocks[2],indexed=True) if indexed else None
    return DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,crop_first=True,
                          deep_projector=projector)


def read_reference(args):
    paths,hashes,plan,traces,costs = inputs(args)
    final = Path(args.final_run)
    phase = Path(args.phase_run)
    old = read(final/'report.json')
    p_contract = read(phase/'contract.json')
    p_report = read(phase/'report.json')
    assert old['status']=='COMPLETED' and old['epoch']==EPOCHS and old['full_budget']
    assert p_report['status']=='COMPLETED' and p_report['exact_scans']==272
    assert p_contract['source_sha256']==SOURCE_SHA and p_contract['inputs_sha256']==hashes
    assert (p_contract['passes'],p_contract['warmup_per_clip'])==(PASSES,WARMUP)
    assert p_report['scenes']==[item[0] for item in plan['scenes']]
    paths.update(final_report=final/'report.json',final_ap=final/'eval_08'/'ap_records.json',
        phase_contract=phase/'contract.json',phase_report=phase/'report.json',phase_equivalence=phase/'equivalence.jsonl')
    hashes = {k:digest(v) for k,v in paths.items()}
    gold = {}
    for line in paths['phase_equivalence'].read_text().splitlines():
        row = json.loads(line)
        key = row['clip'],row['frame']
        assert key not in gold and row['slots']>=row['written']
        assert row['arms']['p5']['core_conv_flops']==costs[key]['p5']
        assert row['arms']['crop']['core_conv_flops']==costs[key]['frozen_local']
        gold[key] = row
    assert set(gold)==set(costs) and len(gold)==272
    return paths,hashes,plan,traces,gold,read(paths['final_ap'])


def verify_mask(masks, online, online_state, fixed_state, stage_conv_ratio):
    written = online_state['deep_age']==0
    assert torch.equal(written,fixed_state['deep_age']==0)
    if masks is None:
        assert online.observation is None and bool(written.all())
        return dict(optional=None,mandatory=None,valid=None,written_cells=int(written.sum()),invalid_cells=0,policy=None)
    assert online.observation is not None
    valid,mandatory,optional = online.observation
    archived_optional = masks['optional']
    assert (optional is None)==(archived_optional is None)
    if optional is not None:
        assert torch.equal(optional,archived_optional) and not bool((mandatory & optional).any())
    archived_guard = masks['written'] if archived_optional is None else masks['written'] & ~archived_optional
    assert torch.equal(valid,masks['valid']) and torch.equal(mandatory,archived_guard)
    assert torch.equal(written,masks['written']) and not bool((~valid & ~mandatory).any())
    assert abs(online.policy.record['planned_sites']/written.numel()-stage_conv_ratio)<1e-12
    return dict(optional=pack_mask(optional),mandatory=pack_mask(mandatory),valid=pack_mask(valid),
                written_cells=int(written.sum()),invalid_cells=int((~valid).sum()),policy=online.policy.record)


@torch.no_grad()
def quality(source,cfg,rows,plan,traces,golden,out,factory):
    models = dict(p5=source,fixed_crop=local_model(source),indexed_online=local_model(source,indexed=True))
    stats = {m:{t:dict(tp=[],fp=[],score=[],gt=0) for t in THRESHOLDS} for m in MODES}
    records,slices,anchors = [],[],{}
    references = dict(p5='p5',fixed_crop='crop',indexed_online='phase_index')
    with (out/'equivalence.jsonl').open('x',buffering=1) as raw:
        for clip,ids,dataset,stream in clip_streams(plan,rows,cfg):
            states = dict.fromkeys(MODES)
            scene,segment = plan['scenes'][clip]
            online = ObservedPolicy(models['indexed_online'],factory)
            start = {m:{t:(len(rec['score']),rec['gt']) for t,rec in value.items()} for m,value in stats.items()}
            for index,sample in enumerate(stream):
                meta = sample['meta']
                key = clip,meta['frame']
                archived = traces[clip][index]
                gold = golden[key]
                assert meta['frame']==ids[index]==archived['frame'] and meta['segment']==segment
                assert bool(meta['supervised'])==archived['supervised'] and archived['refresh']==gold['refresh']
                ego = to_device(sample['ego'],DEVICE)
                spatial = spatial_input(source,ego)
                masks = trace_masks(archived,DEVICE)
                online.observation = None
                requests = dict(p5=None,fixed_crop=local_request(masks),indexed_online=online)
                predictions,measurements,remember = {},{},{}
                for m,model in models.items():
                    (pred,states[m],info),core = measured_step(model,spatial,states[m],meta,requests[m])
                    predictions[m] = pred
                    expected = gold['arms'][references[m]]
                    assert core==expected['core_conv_flops'] and info['full_refresh']==gold['refresh']
                    assert all(bool(torch.isfinite(t).all()) for t in pred.values())
                    if expected['fingerprint'] is not None:
                        remember[m] = fingerprint(pred,states[m])
                        assert remember[m]==expected['fingerprint']
                    model_state = states[m]
                    task_bytes = sum(model_state[k].numel()*model_state[k].element_size() for k in source.state_keys)
                    base = model_state.get('deep_age')
                    phases = model_state.get('deep_projection')
                    measurements[m] = dict(core_conv_flops=core,task_state_bytes=task_bytes,
                        age_bytes=0 if base is None else base.numel()*base.element_size(),
                        projection_bytes=0 if phases is None else phases.numel()*phases.element_size(),
                        stage_conv_ratio=info.get('stage_conv_ratio'))
                if remember:
                    assert set(remember)==set(MODES)
                    anchors[key] = remember
                base,idx = 'fixed_crop','indexed_online'
                assert all(torch.equal(predictions[base][k],predictions[idx][k]) for k in predictions[base])
                assert all(torch.equal(states[base][k],states[idx][k]) for k in (*source.state_keys,'deep_age'))
                mask_report = verify_mask(masks,online,states[idx],states[base],measurements[idx]['stage_conv_ratio'])
                assert mask_report['written_cells']==gold['written'] and mask_report['invalid_cells']==gold['invalid']
                if meta['supervised']:
                    batch = dict(ego=ego)
                    gt = dataset.post.generate_gt_bbx(batch)
                    for m in MODES:
                        score_arm(dataset.post,batch,predictions[m],gt,stats[m])
                row = dict(clip=clip,scene=scene,frame=meta['frame'],supervised=bool(meta['supervised']),refresh=gold['refresh'],
                           mask=mask_report,arms=measurements,head_and_recurrent_exact=True,core_cost_exact=True)
                records.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
            slices.append(dict(scene=scene,clip=clip,arms={m:{t:[start[m][t][0],len(r['score']),r['gt']-start[m][t][1]]
                          for t,r in value.items()} for m,value in stats.items()}))
            write_json(out/'status.json',dict(status='RUNNING',phase='quality',frames=len(records),clips=clip+1))
    return records,stats,slices,anchors


@torch.no_grad()
def timing(source,cfg,rows,plan,traces,anchors,args,out,factory,modes):
    models = dict(p5=source,indexed_fixed=local_model(source,indexed=True),indexed_online=local_model(source,indexed=True))
    if args.cached_residual:
        models['indexed_original'] = local_model(source,indexed=True)
    checks,records = 0,[]
    cache_hits = cache_misses = 0
    with (out/'timing.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(args.uuid)
            checks += 1
            for clip,ids,_,stream in clip_streams(plan,rows,cfg):
                states = dict.fromkeys(modes)
                online = factory(models['indexed_online'])
                if args.cached_residual:
                    old_policy = original_online_policy(models['indexed_original'])
                for i,sample in enumerate(stream):
                    if i%16==0:
                        exclusive(args.uuid)
                        checks += 1
                    meta = sample['meta']
                    archived = traces[clip][i]
                    assert meta['frame']==ids[i]==archived['frame']
                    ego = to_device(sample['ego'],DEVICE)
                    masks = trace_masks(archived,DEVICE)
                    requests = dict(p5=None,indexed_fixed=None if masks is None else masks['optional'],indexed_online=online)
                    if args.cached_residual:
                        requests['indexed_original'] = old_policy
                    shift = (i+clip+repeat)%len(modes)
                    order = modes[shift:] + modes[:shift]
                    elapsed = {}
                    for m in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        spatial = spatial_input(source,ego)
                        pred,states[m],info = models[m].step_spatial(spatial,states[m],meta,requests=requests[m])
                        torch.cuda.synchronize()
                        elapsed[m] = (time.perf_counter()-start)*1000
                        assert info['full_refresh']==archived['refresh']
                        anchor = anchors.get((clip,meta['frame']))
                        if anchor is not None:
                            assert fingerprint(pred,states[m])==anchor['p5' if m=='p5' else 'indexed_online']
                    row = dict(clip=clip,index=i,frame=meta['frame'],repeat=repeat,order=order,included=i>=WARMUP,
                               refresh=archived['refresh'],model_step_ms=elapsed)
                    records.append(row)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
                if args.cached_residual:
                    cache_hits += online.hits
                    cache_misses += online.misses
                write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,clips=clip+1))
    exclusive(args.uuid)
    assert len(records)==PASSES*sum(map(len,plan['clips']))
    active = [r for r in records if r['included']]
    groups = dict(overall=active,ordinary=[r for r in active if r['refresh'] is None])
    def summary(data, mode):
        times = [r['model_step_ms'][mode] for r in data]
        return dict(samples=len(times),mean_ms=mean(times),p95_ms=float(np.percentile(times,95)),
            per_pass_mean_ms=[mean(r['model_step_ms'][mode] for r in data if r['repeat']==i) for i in range(PASSES)])
    return dict(exclusive_checks=checks+1,geometry_cache_hits=cache_hits,geometry_cache_misses=cache_misses,
                metrics={m:{k:summary(v,m) for k,v in groups.items()} for m in modes})


def run(args,out):
    paths,hashes,plan,traces,gold,ap_before = read_reference(args)
    args.kind = 'geometry'
    source,cfg,rows,provenance = build(args)
    assert provenance['split']=='val'
    source.period,source.max_age = 5,.55
    original = model_digest(source)
    factory = CachedResidualPolicy if args.cached_residual else original_online_policy
    modes = (*TIMING_MODES,'indexed_original') if args.cached_residual else TIMING_MODES
    policy = ('Geometry-only LRU keyed by the exact current required mask and lattice shapes; pairs built by '
              'the original planner. Compute the same FP32 residual and FP64 rank in the detector callback.'
              if args.cached_residual else 'Unmodified CausalPolicy residual callback.')
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        inputs_sha256=hashes,source_sha256=SOURCE_SHA,device=torch.cuda.get_device_name(),torch=torch.__version__,
        period=5,max_age=.55,tile=TILE,cap=CAP,arms=MODES,uuid=args.uuid,
        policy=policy+' Archives compare only after the choice.',
        geometry_cache_max_entries=MAX_GEOMETRIES if args.cached_residual else 0,
        timing='Optional exclusive Pillar-to-head no-hooks pass, including online residual/mask generation on indexed online. Indexed fixed uses an external optional mask prepared outside timing; loader/H2D/postprocess excluded for all.',
        timing_arms=modes,repeats=PASSES,warmup_per_scene=WARMUP))
    records,stats,slices,anchors = quality(source,cfg,rows,plan,traces,gold,out,factory)
    assert len(records)==272 and sum(r['supervised'] for r in records)==plan['labelled']
    for m,old in zip(MODES,('p5','frozen_local','frozen_local')):
        assert {str(t):v for t,v in stats[m].items()}==ap_before[old],m
    assert model_digest(source)==original
    write_json(out/'ap_records.json',stats)
    write_json(out/'scene_slices.json',slices)
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,
        orders={m:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in a.items()} for m,a in stats.items()}))
    metrics = {}
    for m in MODES:
        data = [r['arms'][m] for r in records]
        metrics[m] = dict(parameters=sum(p.numel() for p in source.parameters()),ap50=eval_utils.calculate_ap(stats[m],.5)[0],
            ap70=eval_utils.calculate_ap(stats[m],.7)[0],core_conv_flops_mean=mean(d['core_conv_flops'] for d in data),
            maximum_state_bytes={k:max(d[k] for d in data) for k in ('task_state_bytes','age_bytes','projection_bytes')})
    quality_report = dict(status='COMPLETED',scans=len(records),scenes=len(slices),supervised=plan['labelled'],
        gt=stats['p5'][.7]['gt'],raw_ap_exact=True,online_masks_guards_exact=True,indexed_crop_states_predictions_exact=True,
        core_costs_exact=True,ordinary_online_calls=sum(r['refresh'] is None for r in records),metrics=metrics)
    write_json(out/'quality_report.json',quality_report)
    timed = timing(source,cfg,rows,plan,traces,anchors,args,out,factory,modes) if args.uuid else None
    assert model_digest(source)==original and all(digest(paths[k])==h for k,h in hashes.items())
    write_json(out/'report.json',dict(quality_report,timing=timed,timing_valid=timed is not None))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(records),timed=timed is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','final-run','phase-run','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--uuid',help='Optional exclusive device UUID for separate timing')
    parser.add_argument('--cached-residual',action='store_true',help='Reuse geometry-only candidate buckets')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False,
                                         quality_completed=(out/'quality_report.json').is_file()))
        raise


if __name__=='__main__':
    main()

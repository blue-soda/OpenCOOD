"""Run an equal-core causal block-event pilot with actual recurrent writeback."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_block_adapt_transfer import average_precision
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_membrane import score_arm
from opencood.tools.eval_int_deep_block_rank import checked_plan, clip_streams, tile_features
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, group_bucket, masks_by_tile, spatial_input
from opencood.tools.probe_int_deep_block_event_rank import CAP, QUANTILE, RESET, TAU, TILE, lift, tile_reduce, warp_indices, write_json
from opencood.tools.run_int_ego import to_device
from opencood.utils import eval_utils


MODES = ('residual', 'continuous', 'pulse_request_clear', 'pulse_write_clear')
PULSES = MODES[2:]
THRESHOLDS = (.5, .7)
COUNTERS = ('issued','new_requests','carried_requests','granted_requests','served_requests','unserved_requests',
            'early_discarded_requests','latched_after_write','waited_served_requests','refresh_cleared_pending_cells')


def read(path):
    return json.loads(Path(path).read_text())


def read_thresholds(path, plan, plan_sha):
    directory = Path(path)
    meta,report = read(directory/'contract.json'),read(directory/'report.json')
    assert meta['source_sha256']==SOURCE_SHA and meta['input_sha256']['plan']==plan_sha
    assert (meta['tau_s'],meta['threshold_quantile'])==(TAU,QUANTILE)
    assert report['status']=='COMPLETED' and report['inspect_only'] is False
    scenes = {item[0] for item in plan['scenes']}
    thresholds = {}
    for fold in report['folds']:
        held = fold['held_scene']
        assert held in scenes and held not in thresholds and set(fold['training_scenes'])==scenes-{held}
        assert fold['threshold_samples']>0 and math.isfinite(fold['threshold']) and fold['threshold']>0
        thresholds[held] = fold['threshold']
    assert set(thresholds)==scenes
    return thresholds, {n:digest(directory/n) for n in ('contract.json','report.json')}


class LiveControl:
    """Aligned score evidence and, for threshold arms, pending requests and wait clocks."""

    def __init__(self, mode, threshold):
        self.mode = mode
        self.threshold = threshold
        self.clear()

    def clear(self):
        self.evidence = self.pending = self.wait = None

    def align(self, mapping, valid, dt):
        support = valid & (mapping>=0)
        result = np.zeros(valid.shape)
        if self.evidence is not None:
            result[support] = self.evidence.ravel()[mapping[support]]*math.exp(-dt/TAU)
        self.evidence = result
        if self.mode in PULSES:
            pending = np.zeros(valid.shape,dtype=bool)
            wait = np.zeros(valid.shape)
            if self.pending is not None:
                pending[support] = self.pending.ravel()[mapping[support]]
                wait[pending] = self.wait.ravel()[mapping[pending]]+dt
            self.pending,self.wait = pending,wait

    def refresh(self, reason, route):
        counters = dict.fromkeys(COUNTERS,0)
        if self.mode=='residual':
            return counters
        if self.pending is not None:
            counters['refresh_cleared_pending_cells'] = int(self.pending.sum())
        if reason in RESET:
            self.clear()
        else:
            assert reason in ('period','age')
            self.align(*route)
            if self.pending is not None:
                self.pending.fill(False)
                self.wait.fill(0.)
        return counters

    def propose(self, route, current):
        if self.mode=='residual':
            return current,np.zeros(current.shape,dtype=bool),np.zeros(current.shape),np.zeros(current.shape,dtype=bool)
        self.align(*route)
        _,valid,_ = route
        self.evidence += lift(current,valid.shape)*valid
        score = tile_reduce(self.evidence,valid)
        if self.mode=='continuous':
            return score,np.zeros(score.shape,dtype=bool),np.zeros(score.shape),np.zeros(score.shape,dtype=bool)
        carried = tile_reduce(self.pending,valid)>=.5
        waited = tile_reduce(self.wait,valid & self.pending)*carried
        requested = carried | (score>=self.threshold)
        self.pending = lift(requested,valid.shape) & valid
        self.wait = lift(waited,valid.shape)*self.pending
        return score,requested,waited,carried

    def issued(self, requested, eligible, shape):
        issued = np.zeros(requested.shape,dtype=bool)
        for key in eligible:
            issued[key] = requested[key]
        if self.mode=='pulse_request_clear':
            self.consume(lift(issued,shape))
        return issued

    def consume(self, written):
        if self.evidence is not None:
            self.evidence[written] = 0.
        if self.pending is not None:
            self.pending[written] = False
            self.wait[written] = 0.

    def feedback(self, written, valid, issued, pair, carried, waited):
        self.consume(written)
        counters = dict.fromkeys(COUNTERS,0)
        if self.mode not in PULSES:
            return counters,[]
        served = issued & (tile_reduce(written,valid)>=1.)
        waits = waited[served].tolist()
        counters.update(issued=int(issued.sum()),new_requests=int((issued & ~carried).sum()),
            carried_requests=int((issued & carried).sum()),granted_requests=sum(bool(issued[k]) for k in pair),
            served_requests=int(served.sum()),unserved_requests=int((issued & ~served).sum()),
            early_discarded_requests=int((issued & ~served).sum()) if self.mode=='pulse_request_clear' else 0,
            latched_after_write=int((tile_reduce(self.pending,valid)>=.5).sum()),
            waited_served_requests=sum(wait>0. for wait in waits))
        return counters,waits

    def bytes(self):
        return sum(x.nbytes for x in (self.evidence,self.pending,self.wait) if x is not None)


class WriteObserver:
    def __init__(self):
        self.mask = None

    def __call__(self, deep, written):
        self.mask = written
        return deep,deep


def choose_pair(pairs, score, requested):
    if not pairs:
        return ()
    return max(pairs,key=lambda pair:(sum(int(requested[k]) for k in pair),sum(float(score[k]) for k in pair)))


def controller_route(cache, spatial, state, meta):
    scan = SimpleNamespace(prior=state,spatial=spatial,meta=meta)
    mapping,valid = warp_indices(cache,scan,torch,block_lidar_range,warp_bev)
    dt = (int(meta['timestamp_us'])-state['timestamp_us'])/1e6
    assert 0<dt<=.25
    return (np.asarray(mapping,dtype=np.int64).reshape(valid.shape[-2:]),
            valid[0,0].cpu().numpy(),dt),valid


def step(source, models, observers, controls, states, spatial, meta, criterion, target):
    ref = MODES[0]
    reasons = {m:models[m].refresh_reason(states[m],meta) for m in MODES}
    reason = reasons[ref]
    assert all(r==reason for r in reasons.values())
    route,tensor_valid = controller_route(models[ref],spatial,states[ref],meta) if reason not in RESET else (None,None)
    common = pairs = None
    guard,prepared = {},{}
    if reason is None:
        for m in MODES:
            prepared[m] = models[m].prepare(spatial,states[m],meta)
            assert torch.equal(prepared[m].valid,tensor_valid) and abs(prepared[m].dt-route[2])<1e-12
            guard[m] = models[m].mandatory(prepared[m])
        common = torch.stack(list(guard.values())).any(0)
        base,sites,pairs = group_bucket(models[ref],prepared[ref],common.cpu(),CAP)
        expected_sites = sites if pairs else base
        eligible = list(masks_by_tile(common.cpu(),TILE))
        matched = all(torch.equal(common,v) for v in guard.values())
    else:
        expected_sites,matched,eligible = None,True,[]
    predictions, arms = {},{}
    for m in MODES:
        control = controls[m]
        request,chosen,mask = None,(),None
        if reason is None:
            evidence,active,waited,carried = control.propose(route,tile_features(prepared[m])[...,0])
            chosen = choose_pair(pairs,evidence,active)
            issued = control.issued(active,eligible,route[1].shape)
            mask = add_tiles(common,chosen,TILE)

            def check(frame, mandatory):
                assert torch.equal(mandatory,guard[m]) and torch.equal(frame.valid,tensor_valid)
                return mask

            request = check
        else:
            counters = control.refresh(reason,route)
            waits = []
        observers[m].mask = None
        (pred,states[m],info), flops = measured_step(models[m],spatial,states[m],meta,request)
        assert info['full_refresh']==reason
        assert all(bool(torch.isfinite(t).all()) for t in pred.values()) and bool(torch.isfinite(states[m]['deep']).all())
        actual = observers[m].mask
        if mask is not None:
            assert torch.equal(actual,mask) and abs(info['stage_conv_ratio']-expected_sites/mask.numel())<1e-12
            counters,waits = control.feedback(actual[0,0].cpu().numpy(),route[1],issued,chosen,carried,waited)
            actual_indices = actual.flatten().nonzero().flatten().tolist()
        else:
            assert actual is None and not bool(states[m]['deep_age'].any())
            actual_indices = None
        loss = float(criterion(pred,target)) if target is not None else None
        assert loss is None or math.isfinite(loss)
        predictions[m] = pred
        arms[m] = dict(chosen=chosen,written_flat_indices=actual_indices,core_conv_flops=flops,loss=loss,
            mandatory_cells=int(guard[m].sum()) if guard else states[m]['deep_age'].numel(),
            controller_bytes=control.bytes(),counters=counters,served_wait_s=waits,
            detector_state_bytes=sum(states[m][k].numel()*states[m][k].element_size() for k in (*source.state_keys,'deep_age')))
    assert len({a['core_conv_flops'] for a in arms.values()})==1
    geometry = dict(refresh=reason,individual_guard_match=matched,common_union_fallback=not matched,
        planned_output_sites=expected_sites,candidate_pairs=pairs or [],eligible_tiles=len(eligible))
    return predictions,arms,geometry


def mean_loss(records, mode, predicate):
    losses = [r['arms'][mode]['loss'] for r in records if r['supervised'] and predicate(r)]
    return mean(losses) if losses else None


def slice_stats(index, raw, arm, iou):
    first,last,gt = index['arms'][arm][iou]
    return average_precision(np.asarray(raw[iou]['score'][first:last]),np.asarray(raw[iou]['tp'][first:last],dtype=np.int64),gt) if gt else None


def summarize(records, stats, scene_slices, parameters):
    modes = {}
    for mode in MODES:
        arms = [r['arms'][mode] for r in records]
        waits = [v for a in arms for v in a['served_wait_s']]
        modes[mode] = dict(parameters=parameters,ap50=eval_utils.calculate_ap(stats[mode],.5)[0],
            ap70=eval_utils.calculate_ap(stats[mode],.7)[0],core_conv_flops_mean=mean(a['core_conv_flops'] for a in arms),
            maximum_detector_state_bytes=max(a['detector_state_bytes'] for a in arms),
            maximum_controller_bytes=max(a['controller_bytes'] for a in arms),model_step_ms_mean=None,
            mean_loss=mean_loss(records,mode,lambda r:True),
            mean_period_or_next_loss=mean_loss(records,mode,lambda r:r['period_or_next']),
            mean_period_only_loss=mean_loss(records,mode,lambda r:r['refresh']=='period'),
            request_totals={k:sum(a['counters'][k] for a in arms) for k in COUNTERS},
            served_wait_ms_mean=1000*mean(waits) if waits else None)
    scene_reports = []
    for index in scene_slices:
        subset = [r for r in records if r['scene']==index['scene']]
        scene_modes = {}
        for m in MODES:
            waits = [v for row in subset for v in row['arms'][m]['served_wait_s']]
            scene_modes[m] = dict(ap50=slice_stats(index,stats[m],m,.5),ap70=slice_stats(index,stats[m],m,.7),
                loss=mean_loss(subset,m,lambda r:True),period_or_next_loss=mean_loss(subset,m,lambda r:r['period_or_next']),
                request_totals={k:sum(row['arms'][m]['counters'][k] for row in subset) for k in COUNTERS},
                served_wait_ms_mean=1000*mean(waits) if waits else None)
        scene_reports.append(dict(scene=index['scene'],scans=len(subset),gt=index['arms'][MODES[0]][.7][2],modes=scene_modes))
    return modes,scene_reports


@torch.no_grad()
def evaluate(source,cfg,rows,plan,thresholds,out):
    observers = {m:WriteObserver() for m in MODES}
    models = {m:DeepBlockCache(source,TILE,'light',deep_interface=observers[m],crop_first=True,
                              planner=plan_host_coalesced_refresh) for m in MODES}
    stats = {m:{t:dict(tp=[],fp=[],score=[],gt=0) for t in THRESHOLDS} for m in MODES}
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    records,slices = [],[]
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for clip,ids,dataset,stream in clip_streams(plan,rows,cfg):
            scene,segment = plan['scenes'][clip]
            controls = {m:LiveControl(m,thresholds[scene]) for m in MODES if m!='residual'}
            controls['residual'] = LiveControl('residual',None)
            states = dict.fromkeys(MODES)
            start = {m:{t:(len(d['score']),d['gt']) for t,d in a.items()} for m,a in stats.items()}
            last = None
            for index,sample in enumerate(stream):
                meta = sample['meta']
                assert ids[index]==meta['frame'] and segment==meta['segment']
                ego = to_device(sample['ego'],torch.device('cuda'))
                spatial = spatial_input(source,ego)
                target = ego['label_dict'] if meta['supervised'] else None
                predictions,arms,geometry = step(source,models,observers,controls,states,spatial,meta,criterion,target)
                if target is not None:
                    batch = dict(ego=ego)
                    gt = dataset.post.generate_gt_bbx(batch)
                    for m in MODES:
                        score_arm(dataset.post,batch,predictions[m],gt,stats[m])
                refresh = geometry['refresh']
                row = dict(frame=meta['frame'],scene=scene,clip=clip,timestamp_us=int(meta['timestamp_us']),
                    period_or_next=refresh=='period' or (refresh is None and last=='period'),supervised=target is not None,
                    arms=arms,**geometry)
                last = refresh
                raw.write(json.dumps(row,allow_nan=False)+'\n')
                records.append(row)
            slices.append(dict(scene=scene,arms={m:{t:[start[m][t][0],len(r['score']),r['gt']-start[m][t][1]]
                          for t,r in a.items()} for m,a in stats.items()}))
            write_json(out/'status.json',dict(status='RUNNING',scans=len(records),clips=clip+1))
    return records,stats,slices


def run(args,out):
    plan = checked_plan(args.plan)
    plan_sha = digest(Path(args.plan))
    assert plan['split']=='train' and len(plan['clips'])==12 and all(len(c)==20 for c in plan['clips'])
    thresholds,hashes = read_thresholds(args.threshold_run,plan,plan_sha)
    args.kind = 'quality'
    source,cfg,rows,provenance = build(args)
    assert provenance['split']=='train'
    source.period,source.max_age = 5,.55
    original = model_digest(source)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA,plan_sha256=plan_sha,threshold_receipts_sha256=hashes,thresholds=thresholds,
        device=torch.cuda.get_device_name(),torch=torch.__version__,tau_s=TAU,quantile=QUANTILE,
        period=5,max_age=.55,tile=TILE,cap=CAP,arms=MODES,
        geometry='Per-arm causal detector state. Equal-post-halo adjacent-pair bucket on common guard. When guards diverge, their union is a common safety fallback; if the bucket is empty, all perform required writes only.',
        events='Both rank by requested member count then leaky score. Request-clear consumes eligible emitted requests before grant. Write-clear consumes only executed writes. All retain decayed evidence and clear pending at period/age; scene/gap/explicit resets clear both.',
        scope='Real frozen closed-loop paths over training-side development scenes. R299 target-free scene-excluded thresholds are loaded without recalibration. Loss and BEV AP, no training, latency or energy.'))
    records,stats,slices = evaluate(source,cfg,rows,plan,thresholds,out)
    assert [r['frame'] for r in records]==[f for clip in plan['clips'] for f in clip]
    assert model_digest(source)==original and digest(Path(args.plan))==plan_sha
    assert all(digest(Path(args.threshold_run)/n)==h for n,h in hashes.items())
    metrics,per_scene = summarize(records,stats,slices,sum(p.numel() for p in source.parameters()))
    ordinary = [r for r in records if r['refresh'] is None]
    write_json(out/'ap_records.json',stats)
    write_json(out/'scene_slices.json',slices)
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,
        orders={m:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in a.items()} for m,a in stats.items()}))
    write_json(out/'report.json',dict(status='COMPLETED',source_weights_unchanged=True,scans=len(records),
        supervised=sum(r['supervised'] for r in records),gt=stats['residual'][.7]['gt'],scenes=len(slices),
        ordinary_scans=len(ordinary),matched_core_frames=len(records),common_union_fallbacks=sum(r['common_union_fallback'] for r in ordinary),
        mandatory_only_ordinaries=sum(not r['candidate_pairs'] for r in ordinary),
        refresh_reasons=dict(Counter(r['refresh'] or 'ordinary' for r in records)),
        pulse_write_vs_request_different_masks=sum(r['arms'][PULSES[0]]['chosen']!=r['arms'][PULSES[1]]['chosen'] for r in ordinary),
        comparisons_to_residual_changed_pairs={m:sum(r['arms'][m]['chosen']!=r['arms']['residual']['chosen'] for r in ordinary) for m in MODES[1:]},
        metrics=metrics,per_scene=per_scene))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','threshold-run','output'):
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

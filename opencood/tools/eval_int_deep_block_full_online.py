"""R-170-native full validation of online indexed deep reuse and dispatch."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np

from opencood.tools.eval_int_deep_block_long_online import read, selection_equal, write
from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest


SCANS, LABELS, GT = 5794, 1738, 22971
PERIOD, AGE, TILE = 5, .55, 8
SEED = 303
MODES = ('p5', 'indexed_crop_online', 'indexed_dense_online')
CROP, DENSE = MODES[1:]
IOU = ('0.5', '0.7')
REFERENCE_KEYS = ('contract.json', 'frames.jsonl', 'report.json', 'ap_records.json')


def read_r170(directory):
    folder = Path(directory)
    hashes = {name:digest(folder/name) for name in REFERENCE_KEYS}
    contract = read(folder/'contract.json')
    report = read(folder/'report.json')
    frames = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
    raw = read(folder/'ap_records.json')['p5_rank128']
    assert contract['precision']==dict(matmul_tf32=False,cudnn_tf32=False)
    assert report['status']=='COMPLETED' and report['scans']==SCANS and report['labels']==LABELS
    assert contract['frames']==[r['frame'] for r in frames] and len(frames)==SCANS
    assert sum(r['supervised'] for r in frames)==LABELS and all(raw[t]['gt']==GT for t in IOU)
    return contract,hashes,frames,raw


def row_identity(row):
    return {key:row[key] for key in ('frame','scene','segment','timestamp_us','supervised','reset_before')}


def segment_ranges(rows):
    boundaries = [i for i in range(len(rows)) if i == 0 or rows[i]['segment']!=rows[i-1]['segment']]
    return list(zip(boundaries,boundaries[1:]+[len(rows)]))


def native_inputs(root):
    folder = Path(root)/'full_gru'
    runtime = read(folder/'effective_runtime.json')['arguments']
    contract = read(folder/'manifest.json')['contract']
    manifest = Path(artifact_path(runtime['manifest']))
    config = Path(artifact_path(runtime['config']))
    assert digest(manifest)==contract['manifest_sha256'] and digest(config)==contract['config_sha256']
    rows = read(manifest)['val']
    assert len(rows)==SCANS and len({row['frame'] for row in rows})==SCANS
    assert sum(row['supervised'] for row in rows)==LABELS and all(row['split']=='val' for row in rows)
    return rows,contract


def prepare(args,out):
    _,hashes,prior,_ = read_r170(args.r170)
    rows,contract = native_inputs(args.root)
    frozen = [row_identity(r) for r in rows]
    assert [(r['frame'],r['supervised']) for r in frozen]==[(r['frame'],r['supervised']) for r in prior]
    required = sorted({r[k] for r in rows for k in ('pcd','label') if r[k] is not None})
    assert len(required)==SCANS+LABELS
    measurements = {}
    for i,name in enumerate(required):
        path = Path(artifact_path(name))
        measurements[name] = dict(bytes=path.stat().st_size,sha256=digest(path))
        if (i+1)%512==0:
            write(out/'status.json',dict(status='RUNNING',phase='hash_native_inputs',hashed=i+1,total=len(required)))
    plan = dict(split='val',manifest_sha256=contract['manifest_sha256'],config_sha256=contract['config_sha256'],
                r170_sha256=hashes,ordered_stream=frozen,input_files=measurements,
                warmup_excluded=0,seed=SEED,segments=len(segment_ranges(frozen)))
    write(out/'plan.json',plan)
    write(out/'report.json',dict(status='COMPLETED',scans=SCANS,labels=LABELS,segments=plan['segments'],
        scenes=len({r['scene'] for r in rows}),files=len(measurements),plan_sha256=digest(out/'plan.json')))


def check_files(plan,out):
    files = plan['input_files']
    for i,(name,receipt) in enumerate(files.items()):
        path = Path(artifact_path(name))
        assert path.stat().st_size==receipt['bytes'] and digest(path)==receipt['sha256'], name
        if (i+1)%512==0:
            write(out/'status.json',dict(status='RUNNING',phase='verify_native_inputs',hashed=i+1,total=len(files)))


def blank_ap():
    return {name:{t:dict(tp=[],fp=[],score=[],gt=0) for t in IOU} for name in MODES}


def merge_ap(target, part):
    assert target.keys() == part.keys()
    for mode in target:
        for threshold in IOU:
            a,b = target[mode][threshold],part[mode][threshold]
            for key in ('tp','fp','score'):
                a[key].extend(b[key])
            a['gt'] += b['gt']


def committed_segments(folder,ranges,rows,identity):
    previous = list(sorted(folder.glob('*.json')))
    assert [p.name for p in previous]==['%04d.json' % i for i in range(len(previous))]
    assert len(previous)<=len(ranges)
    merged_records,merged_ap = [],blank_ap()
    for i,path in enumerate(previous):
        receipt = read(path)
        payload = receipt['payload']
        assert sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()==receipt['sha256']
        start,end = ranges[i]
        assert payload['identity']==identity and payload['bounds']==[start,end]
        assert [r['frame'] for r in payload['records']]==[r['frame'] for r in rows[start:end]]
        merged_records.extend(payload['records'])
        merge_ap(merged_ap,payload['ap'])
    return len(previous),merged_records,merged_ap


def commit_segment(path,payload):
    partial = path.with_suffix('.partial')
    checksum = sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    write(partial,dict(payload=payload,sha256=checksum))
    os.replace(partial,path)


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
    from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
    from opencood.tools.profile_int_deep_block_cache import PROJECTION_TOLERANCE
    from opencood.tools.profile_int_deep_phase_edges import ObservedPlanner, dense_plan
    from opencood.tools.run_int_ego import loader, to_device
    from opencood.utils import eval_utils

    plan = read(args.plan)
    plan_sha = digest(Path(args.plan))
    reference,r170_hashes,gold,prior_ap = read_r170(args.r170)
    precision = reference['precision']
    torch.backends.cuda.matmul.allow_tf32 = precision['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = precision['cudnn_tf32']
    assert plan['r170_sha256']==r170_hashes and plan['split']=='val' and plan['seed']==SEED and plan['warmup_excluded']==0
    assert reference['checkpoint_sha256']==SOURCE_SHA['checkpoint']
    assert reference['initialization_sha256']==SOURCE_SHA['readout']
    assert reference['protected_trained_sha256']==SOURCE_SHA['trained']
    native,provenance = native_inputs(args.root)
    assert provenance['manifest_sha256']==plan['manifest_sha256'] and provenance['config_sha256']==plan['config_sha256']
    assert [row_identity(r) for r in native]==plan['ordered_stream']
    assert set(plan['input_files'])=={r[k] for r in native for k in ('pcd','label') if r[k] is not None}
    ranges = segment_ranges(native)
    assert len(ranges)==plan['segments']
    key = dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               source_sha256=SOURCE_SHA,plan_sha256=plan_sha,r170_sha256=r170_hashes,period=PERIOD,max_age=AGE,tile=TILE,
               workers=args.workers,seed=SEED,precision=precision,torch=torch.__version__,device=torch.cuda.get_device_name())
    contract = dict(identity=key,arms=list(MODES),scans=SCANS,labelled_scans=LABELS,warmup_excluded=0,
        scope='Complete historical DAIR validation split. Frozen weights. Every labeled scan scored, all unlabeled scans consumed in original native order. AP and Conv/ConvTranspose only.',
        segments='Task states persist across the stream and reset naturally via manifest boundaries. Reuse pure geometry cache within each segment only. Resume uses prior segment identity to preserve next natural reset.',
        geometry='Same online residual policy and written mask, independent recurrent states, indexed projection on both local arms. Direct host crop vs R-303 same-mask dense dispatch.',
        audit='Exact original policy on every ordinary frame. Compare bitwise and separate numerical tolerance. Compare P5 raw TP/FP/score with R-170 only after full-frame identity.',
        tolerance=PROJECTION_TOLERANCE)
    if args.resume:
        assert read(out/'contract.json')==contract
    else:
        write(out/'contract.json',contract)
    segments = out/'segments'
    segments.mkdir(exist_ok=args.resume)
    next_chunk,records,ap = committed_segments(segments,ranges,native,key)
    check_files(plan,out)
    args.kind = 'geometry'
    source,cfg,rows,origin = build(args)
    source.period,source.max_age = PERIOD,AGE
    assert [row_identity(r) for r in rows]==plan['ordered_stream'] and origin['split']=='val'
    frozen = model_digest(source)
    planners = {m:ObservedPlanner(p) for m,p in ((CROP,plan_host_coalesced_refresh),(DENSE,dense_plan))}
    models = {'p5':source}
    for m,p in planners.items():
        models[m] = DeepBlockCache(source,TILE,'light',planner=p,crop_first=True,
            deep_projector=DeepPhaseProjection(source.core.deblocks[2],indexed=True))
    handles = [models[m].stage.register_forward_pre_hook(p.observe) for m,p in planners.items()]
    per_site = sum(2*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                   for m in models[CROP].stage.modules() if isinstance(m,nn.Conv2d))
    offset = ranges[next_chunk][0] if next_chunk<len(ranges) else SCANS
    # Native reset_reason consults this segment identity before any past task tensors.
    states = dict.fromkeys(MODES,None if offset==0 else dict(scene=rows[offset-1]['segment']))
    dataset,dataloader = loader(rows[offset:],cfg,args.workers,SEED) if offset<SCANS else (None,None)
    stream = iter(dataloader) if dataloader is not None else iter(())
    try:
        with torch.no_grad():
            for shard,(lo,hi) in enumerate(ranges[next_chunk:],start=next_chunk):
                policies = {m:ObservedPolicy(models[m],CachedResidualPolicy) for m in planners}
                original = original_online_policy(models[CROP])

                def checked_policy(frame,mandatory):
                    result = policies[CROP](frame,mandatory)
                    reference = original(frame,mandatory)
                    assert selection_equal(original.record,policies[CROP].policy.record)
                    assert (result is None)==(reference is None) and (result is None or torch.equal(result,reference))
                    return result

                sub_rows,sub_ap = [],blank_ap()
                # Use float keys only during live evaluation and serialize using R-170's JSON spelling.
                live_ap = {m:{float(t):v for t,v in sub_ap[m].items()} for m in MODES}
                requests = dict(p5=None,**{CROP:checked_policy,DENSE:policies[DENSE]})
                for i in range(lo,hi):
                    sample = next(stream)
                    meta = sample['meta']
                    assert row_identity(meta)==plan['ordered_stream'][i] and sample['ego'] is not None
                    ego = to_device(sample['ego'],torch.device('cuda'))
                    spatial = spatial_input(source,ego)
                    predicted,arms = {},{}
                    for m in MODES:
                        if m in planners:
                            planners[m].reset()
                            policies[m].observation = None
                        (outcome,states[m],info),cost = measured_step(models[m],spatial,states[m],meta,requests[m])
                        assert all(bool(torch.isfinite(t).all()) for t in outcome.values())
                        predicted[m] = outcome
                        names = source.state_keys if m=='p5' else (*source.state_keys,'deep_age','deep_projection')
                        buffer = states[m]
                        layout = planners[m].record(buffer['deep_age'],info['full_refresh']) if m in planners else None
                        arms[m] = dict(core_conv_flops=cost,state_bytes=sum(buffer[k].numel()*buffer[k].element_size() for k in names),
                                       refresh=info['full_refresh'],reset=info['reset'],layout=layout)
                    prior = gold[i]['arms']['p5_rank128']
                    p5 = arms['p5']
                    assert p5['refresh']==prior['refresh'] and p5['reset']==prior['reset'] and p5['core_conv_flops']==prior['core_conv_flops']
                    assert p5['state_bytes']==prior['cache_bytes'] and all(arm['refresh']==p5['refresh'] for arm in arms.values())
                    invalid,written = 0,None
                    if p5['refresh'] is None:
                        valid,guard,optional = policies[CROP].observation
                        v,g,o = policies[DENSE].observation
                        assert torch.equal(valid,v) and torch.equal(guard,g)
                        assert (optional is None)==(o is None) and (o is None or torch.equal(optional,o))
                        assert selection_equal(policies[CROP].policy.record,policies[DENSE].policy.record)
                        written = guard if optional is None else guard|optional
                        assert not bool((~valid & ~written).any())
                        assert torch.equal(written,states[CROP]['deep_age']==0) and torch.equal(written,states[DENSE]['deep_age']==0)
                        invalid = int((~valid).sum())
                        for m in planners:
                            arms[m]['selection'] = policies[m].policy.record
                    else:
                        assert all(p.observation is None for p in policies.values())
                        assert not any(bool(states[m]['deep_age'].any()) for m in planners)
                    added = arms[DENSE]['layout']['work_sites']-arms[CROP]['layout']['work_sites']
                    assert arms[DENSE]['core_conv_flops']-arms[CROP]['core_conv_flops']==added*per_site
                    pairs = [(states[CROP][k],states[DENSE][k]) for k in (*source.state_keys,'deep_age','deep_projection')]
                    pairs += [(t,predicted[DENSE][k]) for k,t in predicted[CROP].items()]
                    assert all(torch.allclose(a,b,**PROJECTION_TOLERANCE) for a,b in pairs)
                    exact = all(torch.equal(a,b) for a,b in pairs)
                    if meta['supervised']:
                        batch = dict(ego=ego)
                        gt = dataset.post.generate_gt_bbx(batch)
                        for m in MODES:
                            score_arm(dataset.post,batch,predicted[m],gt,live_ap[m])
                    row = dict(index=i,frame=meta['frame'],scene=meta['scene'],segment=meta['segment'],supervised=bool(meta['supervised']),
                        refresh=p5['refresh'],invalid=invalid,written=states[CROP]['deep_age'].numel() if written is None else int(written.sum()),
                        written_indices=None if written is None else written.flatten().nonzero().flatten().tolist(),
                        arms=arms,layout_bitwise=exact,layout_max_abs=0. if exact else max(float((a-b).abs().max()) for a,b in pairs))
                    sub_rows.append(row)
                    if (i+1)%32==0:
                        write(out/'status.json',dict(status='RUNNING',phase='full',scans=i+1,committed=lo,total=SCANS))
                commit_segment(segments/('%04d.json'%shard),dict(identity=key,bounds=[lo,hi],records=sub_rows,ap=sub_ap))
                records.extend(sub_rows)
                merge_ap(ap,sub_ap)
                write(out/'status.json',dict(status='RUNNING',phase='full',scans=hi,committed=hi,total=SCANS))
    finally:
        for handle in handles:
            handle.remove()
    assert next(stream,None) is None and model_digest(source)==frozen
    assert len(records)==SCANS and sum(r['supervised'] for r in records)==LABELS
    assert all(ap[m][t]['gt']==GT for m in MODES for t in IOU)
    assert digest(Path(args.plan))==plan_sha
    assert all(digest(Path(args.r170)/k)==v for k,v in r170_hashes.items())
    raw_path = out/'ap_records.json'
    write(raw_path,ap)
    write(out/'sort_orders.json',dict(source_sha256=digest(raw_path),numpy_version=np.__version__,
        orders={m:{t:np.argsort(-np.asarray(r['score'])).tolist() for t,r in arm.items()} for m,arm in ap.items()}))
    with (out/'frames.jsonl').open('w',buffering=1) as raw:
        for row in records:
            raw.write(json.dumps(row,allow_nan=False)+'\n')
    bitwise = sum(r['layout_bitwise'] for r in records)
    if bitwise==SCANS:
        assert ap[CROP]==ap[DENSE]
    old_ap_exact = ap['p5']==prior_ap
    parameter_count = sum(p.numel() for p in source.parameters())
    ordinary = [r for r in records if r['refresh'] is None]
    def metrics(name):
        states = [r['arms'][name] for r in records]
        costs = lambda rows:mean(r['arms'][name]['core_conv_flops'] for r in rows)
        return dict(parameters=parameter_count,ap50=eval_utils.calculate_ap(ap[name],'0.5')[0],
            ap70=eval_utils.calculate_ap(ap[name],'0.7')[0],model_step_ms_mean=None,
            core_conv_flops_mean=costs(records),ordinary_core_conv_flops_mean=costs(ordinary),
            state_bytes_max=max(r['state_bytes'] for r in states))
    summary = {m:metrics(m) for m in MODES}
    for m in planners:
        summary[m].update(cache_hits=sum(r['arms'][m]['selection']['geometry_cache_hit'] for r in ordinary),
            cache_misses=sum(not r['arms'][m]['selection']['geometry_cache_hit'] for r in ordinary),
            mean_ordinary_calls=mean(r['arms'][m]['layout']['calls'] for r in ordinary))
    prefix = records[:128]
    write(out/'report.json',dict(status='COMPLETED',scans=SCANS,labelled_scans=LABELS,gt=GT,segments=len(ranges),
        scenes=len({r['scene'] for r in records}),r170_p5_raw_ap_exact=old_ap_exact,r170_p5_frame_cost_and_refresh_exact=True,
        layout_bitwise_frames=bitwise,layout_ap_records_exact=ap[CROP]==ap[DENSE],full_write_masks_exact=True,
        original_policy_exact_ordinaries=len(ordinary),actual_edge_ordinary_scans=sum(r['invalid']>0 for r in ordinary),
        actual_edge_scenes=len({r['scene'] for r in ordinary if r['invalid']>0}),
        dense_dispatch_ordinaries=sum(r['arms'][CROP]['layout']['jobs']!=r['arms'][DENSE]['layout']['jobs'] for r in ordinary),
        first_128_audit=dict(bitwise=sum(r['layout_bitwise'] for r in prefix),edge=sum(r['invalid']>0 for r in prefix)),
        layout_max_abs=max(r['layout_max_abs'] for r in records),resumed_segments=next_chunk,
        refresh_reasons=dict(Counter(r['refresh'] or 'ordinary' for r in records)),metrics=summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    for name in ('root','r170','output'):
        parser.add_argument('--'+name,required=True)
    for name in ('plan','checkpoint','trained','readout'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.workers<0 or args.plan_only and args.resume:
        parser.error('Invalid worker count or plan-only resume')
    if not args.plan_only and any(getattr(args,k) is None for k in ('plan','checkpoint','trained','readout')):
        parser.error('Full replay requires a locked plan and the three frozen weights')
    out = Path(args.output)
    if args.resume:
        assert out.is_dir()
    else:
        out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        if args.plan_only:
            prepare(args,out)
        else:
            evaluate(args,out)
        write(out/'status.json',dict(status='COMPLETED',plan_only=args.plan_only,scans=SCANS))
    except Exception as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()

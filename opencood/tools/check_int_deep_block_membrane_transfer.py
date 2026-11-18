"""Replay fixed deep task-state quantizers on the R-292 development scenes."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_block_adapt import TRAINED, tensor_digest
from opencood.tools.check_int_deep_block_adapt_transfer import BOOTSTRAP_SEED, REPLICATES, average_precision
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_membrane import (
    QUANTUM, TaskStateInterface, exact, head_delta, load_quantum, quantized_deep, score_arm,
)
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import ATOL, RTOL, SOURCE_SHA, build, max_delta
from opencood.tools.run_int_ego import write_json
from opencood.tools.train_int_deep_block_adapt import EPOCHS, load_clip, local_request
from opencood.utils import eval_utils


MODES = ('continuous', 'scalar_carry', 'scalar_drop', 'channel_carry', 'channel_drop')
QUANTIZED = MODES[1:]
THRESHOLDS = (.5, .7)
STATE_KEYS = ('value', 'coarse', 'deep', 'deep_age')
CONTRASTS = {
    'scalar_carry_minus_drop': {'scalar_carry': 1, 'scalar_drop': -1},
    'channel_carry_minus_drop': {'channel_carry': 1, 'channel_drop': -1},
    'carry_drop_quantizer_interaction': {'channel_carry': 1, 'channel_drop': -1, 'scalar_carry': -1, 'scalar_drop': 1},
    **{mode + '_minus_continuous':{mode: 1, 'continuous': -1} for mode in QUANTIZED},
}


def read(path):
    return json.loads(Path(path).read_text())


def prepare_reference(args, plan):
    base = Path(args.final_run)
    report = read(base/'report.json')
    assert report['status'] == 'COMPLETED' and report['epoch'] == EPOCHS and report['full_budget']
    contract = read(base/'contract.json')['identity']
    assert contract['source_sha256']==SOURCE_SHA and contract['plans']['val']==digest(Path(args.plan))
    assert (contract['period'],contract['max_age'],contract['tile'])==(5,.55,TILE)
    paths = dict(r292_contract=base/'contract.json', r292_report=base/'report.json',
        traces=base/'fixed_traces.json', prior_frames=base/'eval_08'/'frames.jsonl',
        prior_ap=base/'eval_08'/'ap_records.json', calibration=Path(args.calibration),
        calibration_contract=Path(args.calibration_contract), plan=Path(args.plan))
    hashes = {k:digest(v) for k,v in paths.items()}
    assert hashes['traces']==contract['trace_sha256']
    previous_quantum = read(paths['calibration_contract'])
    assert previous_quantum['checkpoint_sha256']==SOURCE_SHA and previous_quantum['quantum']=='per-channel'
    assert previous_quantum['calibration_sha256']==hashes['calibration']
    traces = read(paths['traces'])['val']
    assert [[row['frame'] for row in clip] for clip in traces]==plan['clips']
    frames = [json.loads(line) for line in paths['prior_frames'].read_text().splitlines()]
    assert [(row['clip'],row['frame'],row['supervised']) for row in frames] == [
        (i,row['frame'],row['supervised']) for i,clip in enumerate(traces) for row in clip]
    return paths,hashes,contract,traces,frames,read(paths['prior_ap'])['frozen_local']


def audit_request(masks, slot):
    if masks is None:
        return None
    request = local_request(masks)

    def capture(frame, mandatory):
        optional = request(frame, mandatory)
        slot.append(frame)
        return optional

    return capture


def audit_state(mode, state, slot, masks, quantum, info):
    if masks is None:
        assert not slot and not bool(state['deep_age'].any())
        written = torch.ones_like(state['deep_age'],dtype=torch.bool)
        invalid = 0
    else:
        assert len(slot)==1 and info['full_refresh'] is None
        frame = slot[0]
        written = masks['written']
        assert not bool((~frame.valid & ~written).any())
        assert torch.equal(state['deep_age'],torch.where(written,torch.zeros_like(frame.age),frame.age))
        assert torch.equal(state['deep'],torch.where(written,state['deep'],frame.aligned))
        assert abs(float(written.float().mean())-info['actual_write_fraction']) < 1e-6
        invalid = int((~frame.valid).sum())
    if mode.endswith('_drop'):
        rounded = quantized_deep(state['deep'], quantum)
        assert not bool((written & (rounded != state['deep'])).any())
    return int(written.sum()),invalid


def explicit_reset(models, states, spatial, meta):
    result = {}
    forced = dict(meta,reset_before=True)
    for name,model in models.items():
        assert states[name] is not None
        warm, reset, info = model.step_spatial(spatial,states[name],forced)
        cold, initial, origin = model.step_spatial(spatial,None,meta)
        assert info['full_refresh']=='explicit' and origin['full_refresh']=='initial'
        exact(warm,cold,warm)
        exact(reset,initial,STATE_KEYS)
        assert not bool(reset['deep_age'].any())
        result[name] = dict(predictions_exact=True,state_exact=True,age_zero=True)
    return dict(frame=meta['frame'],states_from_preceding_scan=True,arms=result)


def bootstrap(stats, slices, metrics, out):
    parts = {}
    for mode,arm in stats.items():
        scores = np.asarray(arm[.7]['score'])
        positive = np.asarray(arm[.7]['tp'],dtype=np.int64)
        bounds = [scene['arms'][mode][.7] for scene in slices]
        assert bounds[0][0]==0 and bounds[-1][1]==len(scores)
        assert all(a[1]==b[0] for a,b in zip(bounds,bounds[1:]))
        parts[mode] = [(scores[lo:hi],positive[lo:hi],gt) for lo,hi,gt in bounds]
    gt = np.asarray([row[2] for row in parts[MODES[0]]])
    assert all([row[2] for row in v]==gt.tolist() and sum(gt)==stats[m][.7]['gt'] for m,v in parts.items())

    def aps(sample):
        runs = {mode:[rows[i] for i in sample] for mode,rows in parts.items()}
        return {mode:average_precision(np.concatenate([r[0] for r in rows]),
                                       np.concatenate([r[1] for r in rows]),sum(r[2] for r in rows))
                for mode,rows in runs.items()}

    baseline = aps(range(len(slices)))
    assert all(abs(baseline[m]-metrics[m]['ap70'])<1e-12 for m in MODES)
    draws = np.random.RandomState(BOOTSTRAP_SEED).randint(len(slices),size=(REPLICATES,len(slices)))
    results = [aps(sample) for sample in draws if gt[sample].sum() > 0]
    distinctions = {}
    for name,coef in CONTRASTS.items():
        point = 100*sum(baseline[m]*w for m,w in coef.items())
        gain = np.array([100*sum(aps_row[m]*w for m,w in coef.items()) for aps_row in results])
        distinctions[name] = dict(estimate_pp=point,percentile95_pp=np.percentile(gain,[2.5,97.5]).tolist(),
                                  positive_fraction=float((gain>0).mean()))
    payload = dict(seed=BOOTSTRAP_SEED,replicates=REPLICATES,valid_replicates=len(results),
        sampling='Paired resampling of complete development scenes; descriptive with six scenes',
        scene_ap70=[dict(scene=row['scene'],gt=int(gt[i]),ap70=aps([i]) if gt[i] else None) for i,row in enumerate(slices)],
        contrasts=distinctions)
    write_json(out/'paired_bootstrap.json',payload)
    return payload


@torch.no_grad()
def evaluate(source, cfg, rows, plan, traces, archived, q, qbytes, out):
    quantums = dict(scalar_carry=QUANTUM,scalar_drop=QUANTUM,channel_carry=q,channel_drop=q)
    interface = {m:TaskStateInterface(m.split('_')[1],quantums[m]) for m in QUANTIZED}
    models = {m:DeepBlockCache(source,TILE,'light',interface.get(m),crop_first=True,planner=plan_host_coalesced_refresh)
              for m in MODES}
    stats = {m:{t:dict(tp=[],fp=[],score=[],gt=0) for t in THRESHOLDS} for m in MODES}
    recorded,slices,reset = [],[],None
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for clip,ids,dataset,stream in clip_streams(plan,rows,cfg):
            scene,segment = plan['scenes'][clip]
            start = {m:{t:(len(r['score']),r['gt']) for t,r in a.items()} for m,a in stats.items()}
            states,dense_states = dict.fromkeys(MODES),dict.fromkeys(QUANTIZED)
            entries = load_clip(ids,stream,traces[clip],source)
            for idx,(meta,ego,spatial,masks) in enumerate(entries):
                gold = archived[len(recorded)]
                trace = traces[clip][idx]
                assert gold['frame']==meta['frame']==trace['frame'] and gold['clip']==clip and meta['segment']==segment
                assert gold['supervised']==bool(meta['supervised'])
                if reset is None and masks is not None:
                    reset = explicit_reset(models,states,spatial,meta)
                    write_json(out/'explicit_reset.json',reset)
                batch = dict(ego=ego)
                gt = dataset.post.generate_gt_bbx(batch) if meta['supervised'] else None
                predictions,metrics = {},{}
                expected = gold['core_conv_flops']['frozen_local']
                identity = None
                for name,model in models.items():
                    slot = []
                    (pred,states[name],info), flops = measured_step(
                        model,spatial,states[name],meta,audit_request(masks,slot))
                    assert flops==expected and info['full_refresh']==trace['refresh']
                    assert all(bool(torch.isfinite(t).all()) for t in pred.values())
                    predictions[name] = pred
                    state = states[name]
                    for key in STATE_KEYS:
                        assert bool(torch.isfinite(state[key]).all())
                    written,invalid = audit_state(name,state,slot,masks,quantums.get(name),info)
                    identity = (written,invalid) if identity is None else identity
                    assert identity==(written,invalid)
                    size = sum(state[k].numel()*state[k].element_size() for k in STATE_KEYS)
                    data = dict(core_conv_flops=flops,state_bytes=size,head_mae_vs_continuous=head_delta(pred,predictions[MODES[0]]))
                    if name in QUANTIZED:
                        data['quantization'] = dict(interface[name].quantization)
                        ref,dense_states[name],_ = model.step_spatial(
                            spatial,dense_states[name],meta,local_request(masks),dense_reference=True)
                        data['dense_prediction_max_abs'] = max_delta(ref,pred,pred)
                        data['dense_state_max_abs'] = max_delta(dense_states[name],state,STATE_KEYS)
                    metrics[name] = data
                    if gt is not None:
                        score_arm(dataset.post,batch,pred,gt,stats[name])
                assert len({m['state_bytes'] for m in metrics.values()})==1
                for name in QUANTIZED:
                    exact(states[MODES[0]],states[name], STATE_KEYS if name.endswith('_carry') else ('value','coarse','deep_age'))
                if trace['refresh'] in ('initial','explicit','scene','gap'):
                    for carry,drop in (('scalar_carry','scalar_drop'),('channel_carry','channel_drop')):
                        exact(predictions[carry],predictions[drop],predictions[carry])
                row = dict(frame=meta['frame'],clip=clip,scene=scene,refresh=trace['refresh'],supervised=bool(meta['supervised']),
                    written_cells=identity[0],invalid_cells=identity[1],fixed_trace_exact=True,off_mask_state_and_age_exact=True,arms=metrics,
                    scalar_carry_drop_state_mae=float((states['scalar_carry']['deep']-states['scalar_drop']['deep']).abs().mean()),
                    channel_carry_drop_state_mae=float((states['channel_carry']['deep']-states['channel_drop']['deep']).abs().mean()))
                recorded.append(row)
                raw.write(json.dumps(row,allow_nan=False)+'\n')
            slices.append(dict(scene=scene,clip=clip,arms={m:{t:[start[m][t][0],len(r['score']),r['gt']-start[m][t][1]]
                          for t,r in a.items()} for m,a in stats.items()}))
            write_json(out/'status.json',dict(status='RUNNING',scans=len(recorded),clips=clip+1))
    assert reset is not None and len(recorded)==len(archived)
    assert sum(r['supervised'] for r in recorded)==plan['labelled']
    metrics = {}
    for name in MODES:
        items = [r['arms'][name] for r in recorded]
        metrics[name] = dict(ap50=eval_utils.calculate_ap(stats[name],.5)[0],ap70=eval_utils.calculate_ap(stats[name],.7)[0],
            parameters=sum(p.numel() for p in source.parameters()),model_step_ms_mean=None,
            core_conv_flops_mean=mean(r['core_conv_flops'] for r in items),state_bytes_max=max(r['state_bytes'] for r in items),
            static_calibration_bytes=qbytes if name.startswith('channel') else 0,
            head_mae_vs_continuous={k:mean(r['head_mae_vs_continuous'][k] for r in items) for k in items[0]['head_mae_vs_continuous']})
        if name in QUANTIZED:
            metrics[name].update(mean_readout_quantization_abs=mean(r['quantization']['mean_abs'] for r in items),
                mean_readout_zero_fraction=mean(r['quantization']['zero_fraction'] for r in items),
                dense_prediction_max_abs=max(r['dense_prediction_max_abs'] for r in items),
                dense_state_max_abs=max(r['dense_state_max_abs'] for r in items),
                exact_dense_frames=sum(r['dense_prediction_max_abs']==r['dense_state_max_abs']==0. for r in items))
    return stats,slices,recorded,metrics


def run(args,out):
    plan = checked_plan(args.plan)
    assert plan['split']=='val' and len(plan['clips'])==len({s[0] for s in plan['scenes']})==6
    assert sum(map(len,plan['clips']))==272
    paths,hashes,reference,traces,archived,old_ap = prepare_reference(args,plan)
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    original = model_digest(source)
    assert tensor_digest(source.state_dict(),TRAINED)==reference['frozen_digest']
    q,qsha,qbytes = load_quantum(args.calibration,plan,source.core.deep_light[-1].out_channels)
    assert qsha==hashes['calibration'] and qbytes==2048
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA,inputs_sha256=hashes,source_frozen_digest=original,torch=torch.__version__,
        device=torch.cuda.get_device_name(),period=5,max_age=.55,tile=TILE,arms=MODES,scalar_quantum=QUANTUM,
        per_channel_static_bytes=qbytes,bootstrap_replicates=REPLICATES,bootstrap_seed=BOOTSTRAP_SEED,
        scope='Frozen source; no adapted gate weights. Same R-292 development traces, state buffers and core Conv/ConvTranspose. Readout-only carry rounding; drop rounds the actual written state. No training/timing/energy.',
        dense_reference='Sustained same-mask dense reference for each of the four quantized arms on every frame; max abs and exact frame counts reported separately after tolerance check.',
        dense_tolerance=dict(atol=ATOL,rtol=RTOL)))
    stats,slices,rows,metrics = evaluate(source,cfg,rows,plan,traces,archived,q,qbytes,out)
    assert {str(k):v for k,v in stats['continuous'].items()}==old_ap, 'Original frozen_local TP/FP/score mismatch'
    assert model_digest(source)==original and all(digest(paths[k])==h for k,h in hashes.items())
    write_json(out/'ap_records.json',stats)
    write_json(out/'scene_slices.json',slices)
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,
        orders={m:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in a.items()} for m,a in stats.items()}))
    paired = bootstrap(stats,slices,metrics,out)
    ordinary = [r for r in rows if r['refresh'] is None]
    write_json(out/'report.json',dict(status='COMPLETED',frozen_local_detections_exact=True,weights_unchanged=True,
        scenes=len(slices),scans=len(rows),supervised_scans=sum(r['supervised'] for r in rows),gt=stats['continuous'][.5]['gt'],
        mask_cost_exact_frames=len(rows),explicit_reset=True,dense_reference_frames_each=len(rows),
        ordinary_scans=len(ordinary),ordinary_scans_with_invalid_cells=sum(r['invalid_cells']>0 for r in ordinary),
        ordinary_carry_drop_state_mae={tag:mean(r[tag+'_carry_drop_state_mae'] for r in ordinary) for tag in ('scalar','channel')},
        metrics=metrics,bootstrap=paired))
    write_json(out/'status.json',dict(status='COMPLETED',scans=len(rows),scenes=len(slices)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','final-run','calibration','calibration-contract','output'):
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

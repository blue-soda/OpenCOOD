"""Replay a fixed write trajectory with alternative deep task-state interfaces."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_subthreshold_membrane import quantize_ste
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, max_delta, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.utils import eval_utils


MODES = ('ann', 'continuous', 'carry', 'drop')
QUANTUM = .25
TRACE_SHA = 'e98c16a9aa99bd3db874161e2cc734d2b6e6681c6e33f13144582d14c6c51a82'


class TaskStateInterface:
    def __init__(self, mode):
        self.mode = mode
        self.quantization = None

    def __call__(self, deep, written):
        if self.mode == 'continuous':
            return deep, deep
        sent = quantize_ste(deep, QUANTUM)
        self.quantization = dict(mean_abs=float((deep - sent).abs().mean()),
                                 zero_fraction=float((sent == 0).float().mean()))
        stored = deep
        if self.mode == 'drop':
            stored = sent if written is None else torch.where(written, sent, deep)
        return stored, sent


def exact(left, right, keys):
    assert all(torch.equal(left[k], right[k]) for k in keys)


def head_delta(a, b):
    return {key: float((value - b[key]).abs().mean()) for key, value in a.items()}


def trace_request(tiles, expected_mask):
    def request(frame, mandatory):
        optional = add_tiles(torch.zeros_like(mandatory), tiles, TILE) if tiles else None
        expected_mask.append(mandatory if optional is None else mandatory | optional)
        return optional
    return request


def score_arm(post, batch, pred, gt, stats):
    boxes, scores = post.post_process(batch, dict(ego=pred))
    for t in (.5, .7):
        eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, t)


def run(args, out):
    plan = checked_plan(args.plan)
    assert plan['split'] == 'val'
    trace_sha = digest(Path(args.trace))
    assert trace_sha == TRACE_SHA, 'Unexpected R-275 trace identity'
    trace = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    assert [r['frame'] for r in trace] == [f for clip in plan['clips'] for f in clip]
    args.kind = 'geometry'
    source, cfg, rows, _ = build(args)
    source.period, source.max_age = 5, .55
    original = model_digest(source)
    modes = {mode:TaskStateInterface(mode) for mode in MODES if mode != 'ann'}
    models = {mode:DeepBlockCache(source, TILE, 'light', modes.get(mode)) for mode in MODES}
    state_keys = (*source.state_keys, 'deep_age')
    stats = {m:{t:dict(tp=[], fp=[], score=[], gt=0) for t in (.5,.7)} for m in MODES} if args.score else {}
    write_json(out/'contract.json', dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        checkpoint_sha256=SOURCE_SHA, plan_sha256=digest(Path(args.plan)), fixed_residual_trace_sha256=trace_sha,
        device=torch.cuda.get_device_name(), torch=torch.__version__, period=5, max_age=.55, tile=TILE, quantum=QUANTUM,
        scored=args.score, same_mask_dense_reference=not args.score, scope='Frozen learned P5 dynamics; only deep state storage/readout changed. Carry preserves full pre-quantized task tensor, drop quantizes actual write cells, both round readout. Warp-only retention uses common alignment. No spike transport or timing.'))
    recorded = []
    with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as writer:
        for clip, ids, dataset, stream in clip_streams(plan, rows, cfg):
            state = dict.fromkeys(MODES)
            refs = dict.fromkeys(('carry','drop'))
            for index, sample in enumerate(stream):
                meta = sample['meta']
                record = trace[len(recorded)]
                assert meta['frame'] == ids[index] == record['frame'] and clip == record['clip']
                ego = to_device(sample['ego'], torch.device('cuda'))
                spatial = spatial_input(source, ego)
                gold = record['arms']['residual']
                tiles = gold['schedule']['chosen'] if gold['schedule'] else None
                batch = dict(ego=ego)
                gt = dataset.post.generate_gt_bbx(batch) if args.score and meta['supervised'] else None
                observed, predictions, metrics = {}, {}, {}
                for name, model in models.items():
                    masks = []
                    request = trace_request(tiles, masks)
                    (pred, state[name], info), flops = measured_step(model, spatial, state[name], meta, request)
                    assert flops == gold['core_conv_flops'] and info['full_refresh'] == gold['refresh']
                    assert len(masks) == (0 if gold['refresh'] else 1)
                    assert all(bool(torch.isfinite(x).all()) for x in pred.values())
                    observed[name] = masks[0] if masks else torch.ones_like(state[name]['deep_age'], dtype=torch.bool)
                    if masks:
                        assert abs(float(observed[name].float().mean())-info['actual_write_fraction']) < 1e-6
                        if gold['schedule']:
                            assert abs(info['stage_conv_ratio']-gold['schedule']['planned_sites']/observed[name].numel()) < 1e-12
                    predictions[name] = pred
                    tensors = [state[name][k] for k in state_keys]
                    state_bytes = sum(t.numel()*t.element_size() for t in tensors)
                    assert state_bytes == gold['state_bytes']
                    step = dict(core_conv_flops=flops, state_bytes=state_bytes,
                                quantization=modes[name].quantization if name in modes else None,
                                head_mae_vs_ann=head_delta(pred, predictions['ann']))
                    if name in refs and not args.score:
                        agreement = []
                        reference, refs[name], _ = model.step_spatial(
                            spatial, refs[name], meta, trace_request(tiles, agreement), dense_reference=True)
                        step['dense_head_max_abs'] = max_delta(reference, pred, pred)
                        step['dense_state_max_abs'] = max_delta(refs[name], state[name], state_keys)
                        if masks:
                            exact(dict(mask=observed[name]), dict(mask=agreement[0]), ('mask',))
                    metrics[name] = step
                    if gt is not None:
                        score_arm(dataset.post, batch, pred, gt, stats[name])
                exact(predictions['ann'], predictions['continuous'], predictions['ann'])
                exact(state['ann'], state['continuous'], state_keys)
                for name in MODES[1:]:
                    exact(observed, dict.fromkeys(MODES, observed['ann']), (name,))
                    exact(state['ann'], state[name], ('value','coarse','deep_age'))
                if index == 0:
                    exact(state['ann'], state['carry'], state_keys)
                    exact(predictions['carry'], predictions['drop'], predictions['carry'])
                mask = observed['ann']
                off_mask = (~mask).expand_as(state['drop']['deep'])
                residual = state['drop']['deep'][mask.expand_as(state['drop']['deep'])]
                assert torch.equal(residual, quantize_ste(residual, QUANTUM))
                drift = (state['carry']['deep']-state['drop']['deep']).abs()
                row = dict(frame=meta['frame'], clip=clip, refresh=gold['refresh'], supervised=bool(meta['supervised']),
                           written_cells=int(mask.sum()), lattice_cells=mask.numel(), arms=metrics,
                           carry_drop_state_mae=float(drift.mean()),
                           carry_drop_unwritten_mae=float(drift[off_mask].mean()) if bool(off_mask.any()) else None,
                           carry_drop_head_mae=head_delta(predictions['carry'], predictions['drop']))
                writer.write(json.dumps(row, allow_nan=False)+'\n')
                recorded.append(row)
            write_json(out/'status.json', dict(status='RUNNING', scans=len(recorded), clips=clip+1))
    assert len(recorded) == len(trace) and model_digest(source) == original
    if args.score:
        assert stats['ann'] == stats['continuous']
        write_json(out/'ap_records.json', stats)
        write_json(out/'sort_orders.json', dict(source_sha256=digest(out/'ap_records.json'), numpy_version=np.__version__,
            orders={m:{str(t):np.argsort(-np.asarray(v['score'])).tolist() for t,v in stats[m].items()} for m in MODES}))
    report = {}
    for name in MODES:
        items = [r['arms'][name] for r in recorded]
        report[name] = dict(parameters=sum(p.numel() for p in source.parameters()), model_step_ms_mean=None,
                            core_conv_flops_mean=mean(r['core_conv_flops'] for r in items),
                            state_bytes_max=max(r['state_bytes'] for r in items),
                            ap50=eval_utils.calculate_ap(stats[name], .5)[0] if args.score else None,
                            ap70=eval_utils.calculate_ap(stats[name], .7)[0] if args.score else None,
                            mean_head_mae_vs_ann={key:mean(r['head_mae_vs_ann'][key] for r in items)
                                                  for key in items[0]['head_mae_vs_ann']})
        if name in ('carry','drop'):
            report[name]['mean_readout_quantization_abs'] = mean(r['quantization']['mean_abs'] for r in items)
            report[name]['mean_readout_zero_fraction'] = mean(r['quantization']['zero_fraction'] for r in items)
            if not args.score:
                report[name]['dense_head_max_abs'] = max(r['dense_head_max_abs'] for r in items)
                report[name]['dense_state_max_abs'] = max(r['dense_state_max_abs'] for r in items)
    ordinary = [r for r in recorded if r['refresh'] is None]
    write_json(out/'report.json', dict(status='COMPLETED', scans=len(recorded), weight_unchanged=True,
        exact_mask_cost_frames=len(recorded), exact_ann_continuous_frames=len(recorded),
        initial_scans_equal_carry_drop_head=sum(r['refresh']=='initial' for r in recorded),
        supervised_scans=sum(r['supervised'] for r in recorded) if args.score else None,
        gt=stats['ann'][.5]['gt'] if args.score else None, metrics=report,
        ordinary_carry_drop_state_mae=mean(r['carry_drop_state_mae'] for r in ordinary),
        ordinary_unwritten_carry_drop_state_mae=mean(r['carry_drop_unwritten_mae'] for r in ordinary
                                                    if r['carry_drop_unwritten_mae'] is not None),
        ordinary_carry_drop_head_mae={key:mean(r['carry_drop_head_mae'][key] for r in ordinary)
                                     for key in ordinary[0]['carry_drop_head_mae']}))
    write_json(out/'status.json', dict(status='COMPLETED', scans=len(recorded), scored=args.score))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','trace','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--score', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json', dict(status='RUNNING', phase='inputs'))
    try:
        run(args, out)
    except Exception as exc:
        write_json(out/'status.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

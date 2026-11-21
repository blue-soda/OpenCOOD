"""Calibrate native traffic, fit matched regression heads, or score the paired threshold intervention."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_compact_middle_update import conv_costs
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
from opencood.models.sub_modules.int_silent_receiver import EdgeView, PairedEdge
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy
from opencood.tools.eval_int_fusion_cache import complete_scenes
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.eval_int_online_half import build
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import compact_group_bucket, spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json


def training_rows(base, book_path, calibration):
    folder = base / 'supervised_trace'
    plan, report = read(folder/'plan.json'), read(folder/'report.json')
    assert report['plan'] == digest(folder/'plan.json') and report['frames'] == digest(folder/'frames.jsonl')
    assert all(digest(Path(k)) == v for k, v in plan['members'].items()) and digest(book_path) == plan['book']
    rows = plan['rows']
    if calibration:
        rows = [dict(r, supervised=False, label=None) for r in rows]
    return rows, list(map(json.loads, (folder/'frames.jsonl').read_bytes().splitlines())), report


@torch.no_grad()
def run(args):
    base = args.long_base
    args.inputs = base/'inputs'
    book_path = base/'book_train/report.json'
    book = read(book_path)
    source, cfg, short, _, _, _, quantum = prepare_stream(base/'inputs', base/'clips')
    calibration = args.mode == 'calibration'
    if calibration:
        thresholds = [6, 12]
    else:
        calibration_report = read(args.calibration/'report.json')
        assert calibration_report['status'] == 'COMPLETED' and calibration_report['scans'] == 211
        thresholds = [calibration_report['threshold']]
    paired = PairedEdge(source.bounds, book['cap']//2, book['levels'], thresholds, calibration, args.native)
    names = list(paired.arms)
    paths = names[:1] if calibration else names
    core = source.core
    fusion = core.readout.layers[0].double_conv[0]
    layer = NonzeroFusionCache(fusion[0]).eval()
    fusion[0] = layer
    views = {n: EdgeView(paired, n) for n in paths}
    models = {n: build(source, quantum, views[n]) for n in paths}
    costs = conv_costs(core)
    if args.mode == 'dev':
        rows, traces, identity = complete_scenes(args, short, remainder=args.remainder)
    else:
        rows, traces, identity = training_rows(base, book_path, calibration)
    heads = {n: copy.deepcopy(core.reg_head).eval() for n in paths} if not calibration else {}
    for head in heads.values():
        head.requires_grad_(True)
    fit_hash = None
    if args.mode == 'dev':
        fit_report = read(args.fit/'report.json')
        assert fit_report['steps_per_arm'] == 168 and fit_report['threshold'] == thresholds[0]
        fit_hash = digest(args.fit/'heads.pt')
        assert fit_hash == fit_report['weights']
        params = torch.load(args.fit/'heads.pt', map_location='cpu')
        for n in names:
            heads[n].load_state_dict(params[n])
    opts = {n: torch.optim.Adam(h.parameters(), lr=1e-5) for n,h in heads.items()} if args.mode == 'fit' else {}
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval() if opts else None
    features = []
    if not calibration:
        def capture(module, inputs):
            features.append(inputs[0].detach())
        hook = core.reg_head.register_forward_pre_hook(capture)
    data, stream = loader(rows, cfg, 0, 303)
    write_json(args.output/'contract.json', dict(mode=args.mode, native_grants=args.native, remainder=args.remainder,
        thresholds=thresholds, seed=303,
        scenes=sorted({r['scene'] for r in rows}), data=identity, book=digest(book_path), fit=fit_hash,
        calibration=None if calibration else digest(args.calibration/'report.json'),
        params_per_arm=0 if not heads else sum(p.numel() for p in heads[paths[0]].parameters()),
        code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    totals = dict.fromkeys(names, 0)
    ordinary = labelled = 0
    with (args.output/'frames.jsonl').open('x', buffering=1) as log:
        for i, (sample, trace) in enumerate(zip(stream, traces)):
            meta = sample['meta']
            assert meta['frame'] == trace['frame']
            if i == 0 or meta['segment'] != rows[i-1]['segment']:
                initial = None if trace['refresh'] == 'initial' else dict(scene='')
                states = dict.fromkeys(paths, initial)
                policies = {n: ObservedPolicy(models[n], partial(CachedResidualPolicy, compiler=compact_group_bucket)) for n in paths}
            ego = to_device(sample['ego'], torch.device('cuda'))
            spatial = spatial_input(source, ego)
            reason = trace['refresh']
            paired.start(meta, spatial.shape[-2:], reason)
            row = dict(frame=meta['frame'], scene=meta['scene'], segment=meta['segment'], refresh=reason, arms={})
            if meta['supervised']:
                labelled += 1
            gt = data.post.generate_gt_bbx(dict(ego=ego)) if args.mode == 'dev' and meta['supervised'] else None
            for n in paths:
                model, prev, policy = models[n], states[n], policies[n]
                assert model.refresh_reason(prev, meta) == reason
                model.coarse_projector.begin(prev, reason)
                layer.begin(prev, reason, True)
                policy.observation = None
                (pred, state, info), cost = measured_step(model, spatial, prev, meta, policy)
                assert info['full_refresh'] == reason
                if reason is None:
                    exe = model.coarse_executor
                    cost += sum(costs[k]*v for k,v in exe.counts.items())
                    exe.inputs = None
                state['fusion_cache'] = layer.new
                states[n] = state
                rec = dict(paired.stats[n], core=cost+layer.ledger['float_conv'])
                if not calibration:
                    assert len(features) == 1
                    x = features.pop()
                    if meta['supervised'] and opts:
                        with torch.enable_grad():
                            opts[n].zero_grad()
                            loss = criterion(dict(pred, rm=heads[n](x)), ego['label_dict'])
                            assert bool(torch.isfinite(loss))
                            loss.backward()
                            opts[n].step()
                            rec['loss'] = float(loss)
                    elif gt is not None:
                        rec['frozen'] = score(data.post, ego, pred, gt, record_matches=True)
                        rec['adapted'] = score(data.post, ego, dict(pred, rm=heads[n](x)), gt, record_matches=True)
                        if args.remainder and n in ('carry','reset'):
                            other = 'reset' if n == 'carry' else 'carry'
                            rec['swapped'] = score(data.post, ego, dict(pred, rm=heads[other](x)), gt, record_matches=True)
                row['arms'][n] = rec
            if calibration:
                row['arms'] = paired.stats
            if reason is None:
                ordinary += 1
                for n in names:
                    totals[n] += paired.stats[n]['addresses']
                if not calibration and not args.native:
                    a, b, c, d = (paired.stats[n] for n in names)
                    assert a['addresses'] == b['addresses'] == c['addresses'] and a['bytes'] == b['bytes']
                    assert 0 <= a['bytes'] - d['bytes'] < 12
            log.write(json.dumps(row)+'\n')
    expected_scans, expected_labels = (211, 0 if calibration else 168)
    if args.mode == 'dev':
        expected_scans, expected_labels = (4780, 1280) if args.remainder else (1014, 458)
    assert i+1 == len(rows) == expected_scans and labelled == expected_labels
    result = dict(status='COMPLETED', scans=len(rows), labelled=labelled, ordinary=ordinary,
                  addresses=totals, steps_per_arm=labelled if opts else 0)
    if calibration:
        target = paired.cap / 2
        result['threshold'] = min(thresholds, key=lambda t: (abs(totals[str(t)]/ordinary-target),t))
    else:
        hook.remove()
        result['threshold'] = thresholds[0]
    if opts:
        torch.save({n: h.state_dict() for n,h in heads.items()}, args.output/'heads.pt')
        result['weights'] = digest(args.output/'heads.pt')
    write_json(args.output/'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['calibration','fit','dev'])
    parser.add_argument('--native', action='store_true', help='each development arm grants its own traffic')
    parser.add_argument('--remainder', action='store_true', help='fixed paired transfer on the other validation scenes')
    for k in ('long-base','output','vehicle','calibration','fit'):
        parser.add_argument('--'+k, type=Path, required=k in ('long-base','output'))
    args = parser.parse_args()
    if args.native and args.mode != 'dev':
        parser.error('native grants are a development control')
    if args.remainder and (args.mode != 'dev' or args.native):
        parser.error('transfer requires paired frozen inference')
    if args.mode != 'calibration' and args.calibration is None:
        parser.error('calibration result required')
    if args.mode == 'dev' and (args.fit is None or args.vehicle is None):
        parser.error('fit and vehicle required')
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)

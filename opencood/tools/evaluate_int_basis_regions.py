"""Real causal spatial activation reuse with explicit executed-area/MAC costs."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCacheDetector
from opencood.models.sub_modules.int_fused_basis import fuse_spatial_basis
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout, SplitReadoutDetector
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a):
    out = Path(a.output); write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for k, v in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None, periodic_last=0, max_age=.25).items():
        setattr(a, k, v)
    base, cfg, val, source = setup(a)
    rb, readout = completed_best(a.readout_run)
    artifacts = []
    for root, sha in ((a.memory_run, '1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'),
                      (a.repair_run, '9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'),
                      (a.basis_run, 'c2e173eeed6dd4019319d1f421f81b2582fcf9687184a98a3091c36b16f3c928')):
        root = Path(root)
        assert json.loads((root/'status.json').read_text())['status'] == 'COMPLETED'
        best = json.loads((root/'best.json').read_text())
        assert digest(best['checkpoint']) == best['sha256'] == sha
        artifacts.append((best, torch.load(best['checkpoint'], map_location='cpu')))
    (mb, memory), (pb, repaired), (sb, spatial) = artifacts
    assert memory['contract']['data'] == repaired['contract']['data'] == spatial['contract']['data'] == source['contract']
    assert memory['contract']['source']['sha256'] == readout['contract']['source_sha256'] == source['checkpoint']['sha256']
    assert repaired['contract']['readout'] == spatial['contract']['readout'] == rb
    assert spatial['contract']['memory'] == mb and spatial['contract']['repair'] == pb and spatial['contract']['rank'] == 9
    dense_report = json.loads((Path(a.basis_run)/'report.json').read_text())
    assert dense_report['status'] == 'COMPLETED'
    for name in ('full', 'cached'):
        assert abs(dense_report['results'][name]['ap70']-dense_report['results']['base_'+name]['ap70']) < 1e-5
    base.shrink_conv = compact_shrink(base.shrink_conv, 96)[0].cuda()
    base.shrink_conv.load_state_dict(readout['readout'], strict=True)
    base.feature_memory = projected_gru(base.feature_memory, 32)[0].cuda()
    base.feature_memory.load_state_dict(memory['memory'], strict=True); base.eval()
    split = SplitCompactReadout(base.shrink_conv, base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair = LightRepair(96, width=32).cuda().eval(); repair.load_state_dict(repaired['repair'], strict=True)
    student = copy.deepcopy(base)
    stage = SpatialBasisStage(base.backbone.resnet.layer0, 9).cuda().eval()
    stage.load_state_dict(spatial['stage'], strict=True); student.backbone.resnet.layer0 = stage
    for m in (base, student, split, repair):
        for p in m.parameters(): p.requires_grad_(False)
    modes = dict(base=SplitReadoutDetector(base, cfg, split, 3, repair),
        dense=SplitReadoutDetector(student, cfg, split, 3, repair),
        selected=SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, confidence_threshold=a.confidence_guard),
        masked_dense=SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, execution='masked_dense', confidence_threshold=a.confidence_guard))
    roots = {name:base if name == 'base' else student for name in modes}
    if a.fused:
        fused = copy.deepcopy(student); fused.backbone.resnet.layer0 = fuse_spatial_basis(stage)
        roots['fused'] = fused
        modes['fused'] = SpatialBasisCacheDetector(fused, cfg, split, repair, a.threshold, confidence_threshold=a.confidence_guard)
    control = SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, period=1, confidence_threshold=a.confidence_guard)
    write_json(out/'contract.json', dict(source=source['checkpoint'], data=source['contract'], memory=mb, readout=rb,
        repair=pb, basis=sb, threshold=a.threshold, confidence_guard=a.confidence_guard, fused_execution=a.fused,
        execution_fusion='Optional inference-only composition of full-rank basis+mix into native3x3. Refresh uses native conv; partial gathers selected neighborhoods and computes selected output rows. Replaces basis/mix weights; unchanged four activation caches, reference/guard and mask. No extra dense spatial basis. Count gathered standard convolution MACs, not basis-factor arithmetic.',
        spatial_period=3, max_full_age_s=.25, deep_period=3,
        selector='Current mature stem16-channel mean descriptor at native grid; 4x4-tile RMS relative change to aligned retained reference; radius4 dilation; all uncovered cells mandatory. No labels or current costly features used.',
        confidence_protection='If enabled, union with previous OWN prediction max-anchor sigmoid >= guard threshold, aligned with the spatial state and expanded radius4. Uses pre-NMS confidence, no teacher/GT/current prediction; logs additional area. This may help via extra area alone, not yet a matched-area attribution.',
        recurrence='Four own post-activation maps plus matching retained descriptor warp together by metric SE2 nearest. Write only selected cells; unselected aligned complete activations persist. Save mixed state at current pose; full refresh period3/age/reset. Independent FM and deep anchor states for each mode; no teacher injections.',
        accounting='Actual pointwise output fraction (including dilation and full refresh), dense-basis area, executed conv+functional-linear MACsx2, same-GPU rotating model mean/P95, retained tensor bytes and peak CUDA allocated memory. Nonconv FLOPs not counted.',
        comparison='Compare selected to its own-approximate-recurrence masked_dense control, rank9 dense cache, and mature lowrank cache. Spatial reuse only is changed; old deep cache remains period3 for every mode. No training.'))
    write_json(out/'provenance.json', source)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'opencood/test', '-p', 'test_int*.py'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout); assert tests.returncode == 0
    arguments = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train = json.loads(Path(arguments['manifest']).read_text())['train']
    states = {k:None for k in modes}; control_state = None; costs = {}; mixing = basis_cost = None
    maximum = control_error = fusion_error = 0.; writes = []
    _, stream = loader(train[:32], cfg, 0, 303)
    with torch.no_grad():
        for sample in stream:
            ego = to_device(sample['ego'], torch.device('cuda')); predictions = {}; details = {}
            for name, mode in modes.items():
                counter = ConvCounter(roots[name], torch.nn.ModuleList([split, repair]))
                previous = states[name]
                try: pred, states[name], info = mode.step(ego, states[name], sample['meta'])
                finally: counter.close()
                flag = bool(info['local_partial']); predictions[name] = pred; details[name] = info
                if flag: assert states[name]['cache']['value'] is previous['cache']['value']
                assert all(torch.isfinite(v).all() for v in pred.values())
                if name in ('base', 'dense'):
                    key = (name, flag)
                    if key in costs: assert costs[key] == counter.total
                    costs[key] = counter.total
                else:
                    cells = info['spatial_cells']
                    mixing = sum(cells*c.mix.weight.numel() for b in stage.blocks for c in (b.conv1, b.conv2))
                    basis_cost = sum(cells*c.basis.weight.numel() for b in stage.blocks for c in (b.conv1, b.conv2))
                    skipped = basis_cost if name == 'fused' else 0
                    if name in ('selected', 'fused') and not info['spatial_refresh']: skipped += mixing
                    if name == 'selected' and not info['spatial_refresh'] and info['spatial_selected_cells'] == 0:
                        skipped += basis_cost
                    assert counter.total == costs[('dense', flag)]-skipped, (name, counter.total, skipped)
            for key in predictions['selected']:
                left, right = predictions['selected'][key], predictions['masked_dense'][key]
                maximum = max(maximum, float((left-right).abs().max()))
                assert torch.allclose(left, right, atol=1e-3, rtol=1e-3), maximum
            assert details['selected']['spatial_selected_cells'] == details['masked_dense']['spatial_selected_cells']
            if a.fused:
                for key in predictions['fused']:
                    fusion_error = max(fusion_error, float((predictions['fused'][key]-predictions['selected'][key]).abs().max()))
                    assert torch.allclose(predictions['fused'][key], predictions['selected'][key], atol=1e-3, rtol=1e-3)
                assert details['fused']['spatial_selected_cells'] == details['selected']['spatial_selected_cells']
            writes.append(details['selected']['spatial_write_fraction'])
            pred, control_state, _ = control.step(ego, control_state, sample['meta'])
            for key in pred:
                control_error = max(control_error, float((pred[key]-predictions['dense'][key]).abs().max()))
                assert torch.allclose(pred[key], predictions['dense'][key], atol=1e-5, rtol=1e-5)
    write_json(out/'engineering.json', dict(status='PASSED', frames=32, selected_vs_masked_dense_max_error=maximum,
        fused_vs_factorized_max_error=fusion_error if a.fused else None,
        factorized_stage_parameters=sum(p.numel() for p in stage.parameters()),
        fused_stage_parameters=sum(p.numel() for p in roots['fused'].backbone.resnet.layer0.parameters()) if a.fused else None,
        period1_vs_dense_max_error=control_error, mean_write_fraction=float(np.mean(writes)),
        constant_hook_macs={str(k):v for k,v in costs.items()}, dense_mix_macs=mixing, dense_basis_macs=basis_cost,
        costs_verified_against_actual_conv_hooks=True, own_immutable_deep_anchors=True))
    states = {k:None for k in modes}; control_state = None
    stats = {k:{t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)} for k in modes}
    metrics = {k:dict(time=[], write=[], mix=[], basis=[], state=[], peak=[], full_age=[], uncovered=[], priority=[], guard_added=[],
                     macs=0, refreshes=0, partials=0) for k in modes}
    dataset, stream = loader(val, cfg, 4, 303); fusion_mask_disagreements = 0
    with torch.no_grad():
        for index, sample in enumerate(stream):
            ego = to_device(sample['ego'], torch.device('cuda')); names = list(modes)
            offset = index % len(names); names = names[offset:]+names[:offset]; counts = {}
            for name in names:
                torch.cuda.synchronize(); before = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter(); pred, states[name], info = modes[name].step(ego, states[name], sample['meta'])
                torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-tick)
                item = metrics[name]
                if index >= 10:
                    item['time'].append(elapsed)
                    item['peak'].append(max(0, torch.cuda.max_memory_allocated()-before))
                assert all(torch.isfinite(v).all() for v in pred.values())
                flag = bool(info['local_partial']); item['partials'] += flag
                macs = costs[('base' if name == 'base' else 'dense', flag)]
                if name == 'fused': macs -= basis_cost
                if name in ('selected', 'fused') and not info['spatial_refresh']:
                    macs -= mixing
                    if name == 'selected' and info['spatial_selected_cells'] == 0: macs -= basis_cost
                    macs += info['spatial_extra_linear_macs']
                item['macs'] += macs
                item['write'].append(info.get('spatial_write_fraction', 1.))
                item['mix'].append(info.get('spatial_mixer_executed_fraction', 1.))
                item['basis'].append(info.get('spatial_basis_executed_fraction', 1.))
                item['state'].append(info['state_bytes']+info['anchor_bytes']+info.get('spatial_state_bytes', 0))
                item['refreshes'] += int(info.get('spatial_refresh', True))
                item['full_age'].append(info.get('spatial_full_age', 0.))
                item['uncovered'].append(info.get('spatial_uncovered_fraction', 0.))
                item['priority'].append(info.get('spatial_priority_fraction', 0.))
                item['guard_added'].append(info.get('spatial_guard_added_fraction', 0.))
                if name in ('selected', 'masked_dense', 'fused'): counts[name] = info['spatial_selected_cells']
                if sample['meta']['supervised']:
                    data = dict(ego=ego); boxes, scores = dataset.post.post_process(data, dict(ego=pred)); gt = dataset.post.generate_gt_bbx(data)
                    for threshold in stats[name]: eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[name], threshold)
            assert counts['selected'] == counts['masked_dense'], 'Selection changed with execution backend'
            if a.fused: fusion_mask_disagreements += int(counts['fused'] != counts['selected'])
            if (index+1) % 200 == 0: write_json(out/'status.json', dict(status='RUNNING', phase='validation', scans=index+1))
    assert index+1 == len(val)
    results = {}
    for name, item in metrics.items():
        results[name] = dict(ap70=eval_utils.calculate_ap(stats[name], .7)[0], ap50=eval_utils.calculate_ap(stats[name], .5)[0],
            mean_ms=float(np.mean(item['time'])), p95_ms=float(np.percentile(item['time'], 95)),
            conv_linear_gflops=2*item['macs']/len(val)/1e9,
            mean_write_fraction=float(np.mean(item['write'])), mean_executed_mix_area=float(np.mean(item['mix'])),
            mean_executed_basis_area=float(np.mean(item['basis'])), mean_state_bytes=float(np.mean(item['state'])),
            max_incremental_cuda_peak_bytes=max(item['peak']), mean_uncovered_fraction=float(np.mean(item['uncovered'])),
            mean_priority_fraction=float(np.mean(item['priority'])), mean_guard_added_fraction=float(np.mean(item['guard_added'])),
            max_spatial_full_age_s=max(item['full_age']), spatial_refresh_frames=item['refreshes'], deep_partial_frames=item['partials'])
    assert abs(results['base']['ap70']-.7555532616222492) < 1e-8
    assert abs(results['dense']['ap70']-sb['cached_ap70']) < 1e-8
    write_json(out/'report.json', dict(status='COMPLETED', scans=len(val), labels=sum(r['supervised'] for r in val), results=results,
        fused_mask_count_disagreement_frames=fusion_mask_disagreements,
        scope='Actual own-recursive four-map spatial cache, whole aligned activation reuse, selected output-column mixing. Fixed heuristic threshold, no training. Dense bases (unless fused), stem/current FM and existing readout/deep cache included. Fused execution composes full-rank kernels and eliminates explicit spatial bases, with same cache/mask semantics. Areas include refresh and dilated requests, no duplicated halo outputs. Conv plus functional-linear MACsx2 excludes nonconv arithmetic. Same GPU rotating wall time includes warp/selection/gather/scatter; excludes IO/H2D/postprocess. State tensor bytes are per-mode; incremental CUDA peaks measure transient allocation above live multi-mode allocations, not standalone process VRAM.'))
    write_json(out/'status.json', dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'readout-run', 'repair-run', 'memory-run', 'basis-run', 'output'): p.add_argument('--'+key, required=True)
    p.add_argument('--threshold', type=float, choices=(.25, .5, .75), required=True)
    p.add_argument('--confidence-guard', type=float, choices=(.1, .3), default=None)
    p.add_argument('--fused', action='store_true', help='Include equivalent composed-kernel selected execution')
    p.add_argument('--gpu', type=int, required=True); p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = Path(a.output)
    if a.worker:
        def expired(s, f): raise TimeoutError('Basis region evaluation exceeded4 hours')
        signal.signal(signal.SIGALRM, expired); signal.alarm(14400)
        try: worker(a)
        except Exception as exc: write_json(out/'status.json', dict(status='FAILED', error=repr(exc))); raise
        finally: signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free) < 22000 or int(util) > 5: raise RuntimeError('GPU occupied')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'opencood.tools.evaluate_int_basis_regions']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=proc.pid, gpu=a.gpu, command=command, created_unix=time.time(), timeout_seconds=14400,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid, output=str(out))))


if __name__ == '__main__': main()

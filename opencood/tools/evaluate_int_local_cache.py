"""Paired fixed-weight local-state region reuse evaluation on one GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.models.sub_modules.int_local_state_cache import RegionalLocalStateDetector
from opencood.models.sub_modules.int_local_selector import LocalChangeSelector, selector_state
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils
from opencood.tools.audit_int_local_compute import same_lattice_conv_macs


def make_model(teacher, cfg, saved, **policy):
    weights = dict(saved['model'])
    if policy.get('selector') == 'learned':
        net = LocalChangeSelector()
        net.load_state_dict(policy.pop('selector_weights'), strict=True)
        policy['selector_net'] = net
        weights.update({'selector_net.'+k:v for k,v in net.state_dict().items()})
    model = (RegionalLocalStateDetector(teacher, cfg, **policy) if policy else
             LocalStateDetector(teacher, cfg, 'recurrent')).cuda().eval()
    model.load_state_dict(weights, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def probe(cfg, models, rows):
    states = [None]*4
    errors = [0., 0.]
    partial = 0
    _, stream = loader(rows[:64], cfg, 2, 303)
    with torch.no_grad():
        for sample in stream:
            ego = to_device(sample['ego'], torch.device('cuda'))
            outputs = []
            for i, model in enumerate(models):
                output, states[i], info = model.step(ego, states[i], sample['meta'])
                outputs.append(output)
                assert all(torch.isfinite(v).all() for v in output.values())
                if i == 2:
                    partial += int(0 < info['region_cache']['actual_write_fraction'] < 1)
            for j, (left, right) in enumerate(((0, 1), (2, 3))):
                for key in outputs[left]:
                    error = float((outputs[left][key]-outputs[right][key]).abs().max())
                    errors[j] = max(errors[j], error)
                    assert torch.allclose(outputs[left][key], outputs[right][key], atol=1e-4, rtol=1e-4), (
                        'Recursive engineering mismatch', j, key, error)
            assert torch.equal(states[2]['cache_age'], states[3]['cache_age'])
    return dict(status='PASSED', split='train', frames=min(64, len(rows)),
                dense_all_update_max_error=errors[0], sparse_masked_dense_max_error=errors[1],
                partial_update_frames=partial, teacher_execution_forbidden=True,
                caveat='Masked-dense is a correctness reference, not a compute-saving implementation.')


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    evaluation_stage = a.stage
    for key, value in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None,
                           periodic_last=0, max_age=.25, oracle_stage0=False, oracle_index=0,
                           support_profile=False, zero_uncovered=False).items():
        setattr(a, key, value)
    teacher, cfg, val, provenance = setup(a)
    source = Path(a.checkpoint)
    assert digest(source) == a.source_sha256, 'Source checkpoint hash mismatch'
    saved = torch.load(str(source), map_location='cpu')
    contract = saved['contract']
    assert contract['version'] == LocalStateDetector.version and contract['phase'] == 'b'
    assert contract['memory_policy'] == 'recurrent'
    assert contract['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert contract['data_contract'] == provenance['contract']
    runtime = json.loads((Path(a.root)/'full_gru/effective_runtime.json').read_text())['arguments']
    train = json.loads(Path(runtime['manifest']).read_text())['train']
    source_ap = json.loads((source.parent/'report.json').read_text())['metrics']['bev_ap70']
    provenance.update(evaluated_checkpoint=str(source), evaluated_checkpoint_sha256=a.source_sha256,
                      source_ap70=source_ap, stage=evaluation_stage,
                      policy=dict(fractions=a.fractions, tile=a.tile, max_age=a.cache_max_age,
                                  fallback_fraction=a.fallback_fraction,
                                  selector=a.selector,
                                  mandatory='new_coverage_and_per_cell_age',
                                  execution=a.execution, batch_size=a.batch_size,
                                  fallback_planner='merged_rectangles', radius=4))
    write_json(out/'provenance.json', provenance)
    # Build every copied module BEFORE installing teacher-only hooks: deepcopy
    # also copies hooks, which would otherwise forbid legitimate student calls.
    common = dict(tile=a.tile, max_age=a.cache_max_age, fallback_fraction=a.fallback_fraction,
                  selector=a.selector)
    selector_weights = None
    if a.selector == 'learned':
        assert digest(a.selector_checkpoint) == a.selector_sha256
        selector_saved = torch.load(a.selector_checkpoint, map_location='cpu')
        assert len(a.fractions) == 1, 'Evaluate each scorer at its trained budget'
        assert selector_saved['max_age'] == a.cache_max_age
        assert selector_saved['fallback_fraction'] == a.fallback_fraction
        selector_weights = selector_state(selector_saved, a.source_sha256, a.tile, a.fractions[0])
        common['selector_weights'] = selector_weights
        provenance.update(selector_checkpoint=a.selector_checkpoint, selector_sha256=a.selector_sha256,
                          selector_epoch=selector_saved['epoch'],
                          selector_parameters=sum(v.numel() for v in selector_weights.values()))
        write_json(out/'provenance.json', provenance)
    probe_models = [
        make_model(teacher, cfg, saved),
        make_model(teacher, cfg, saved, fraction=1., **common),
        make_model(teacher, cfg, saved, fraction=min(a.fractions),
                   execution=a.execution, batch_size=a.batch_size, **common),
        make_model(teacher, cfg, saved, fraction=min(a.fractions), execution='masked_dense', **common)]
    models = {'dense': make_model(teacher, cfg, saved)}
    for fraction in a.fractions:
        models['cache_%g'%fraction] = make_model(
            teacher, cfg, saved, fraction=fraction, execution=a.execution, batch_size=a.batch_size, **common)
    def reject_teacher(module, inputs):
        raise RuntimeError('Teacher inference is forbidden')
    hooks = [m.register_forward_pre_hook(reject_teacher) for m in
             (teacher.pillar_vfe, teacher.backbone, teacher.cls_head, teacher.reg_head)]
    try:
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                                'opencood/test', '-p', 'test_int*.py'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out/'unit_tests.log').write_bytes(tests.stdout)
        assert tests.returncode == 0, 'INT tests failed'
        write_json(out/'status.json', dict(status='RUNNING', phase='engineering'))
        engineering = probe(cfg, probe_models, train)
        write_json(out/'engineering.json', engineering)
        del probe_models
        states = {name:None for name in models}
        aps = {name:{t:dict(tp=[], fp=[], gt=0, score=[]) for t in [.3,.5,.7]} for name in models}
        records = {name:[] for name in models}
        chosen = val if evaluation_stage == 'full' else val[:a.frames]
        dataset, stream = loader(chosen, cfg, 4, 303)
        names = list(models)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as log:
            for index, sample in enumerate(stream):
                meta = sample['meta']
                assert meta['frame'] == chosen[index]['frame']
                ego = to_device(sample['ego'], torch.device('cuda'))
                # Rotate first/last position while preserving each model's own
                # causal stream. All costs, including CPU crop planning, count.
                order = names[index % len(names):]+names[:index % len(names)]
                predictions = {}
                for name in order:
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    predictions[name], states[name], info = models[name].step(ego, states[name], meta)
                    torch.cuda.synchronize(); ms = (time.perf_counter()-tick)*1000
                    assert all(torch.isfinite(v).all() for v in predictions[name].values())
                    entry = dict(frame=meta['frame'], mode=name, ms=ms, reset=info['reset'],
                                 state_bytes=info['state_bytes'], region=info.get('region_cache'))
                    records[name].append(entry)
                    log.write(json.dumps(entry)+'\n')
                if meta['supervised']:
                    data = dict(ego=ego)
                    gt = dataset.post.generate_gt_bbx(data)
                    for name, prediction in predictions.items():
                        boxes, scores = dataset.post.post_process(data, dict(ego=prediction))
                        for threshold in aps[name]:
                            eval_utils.caluclate_tp_fp(boxes, scores, gt, aps[name], threshold)
                if (index+1) % 100 == 0:
                    progress = dict(status='RUNNING', phase=evaluation_stage,
                                    scans=index+1, total_scans=len(chosen))
                    write_json(out/'status.json', progress)
                    print('EVAL', json.dumps(progress), flush=True)
        assert all(len(values) == len(chosen) for values in records.values())
        results = {}
        for name, model in models.items():
            expected = dict(saved['model'])
            if name != 'dense' and selector_weights is not None:
                expected.update({'selector_net.'+k:v for k,v in selector_weights.items()})
            assert set(expected) == set(model.state_dict())
            assert all(torch.equal(v.cpu(), expected[k]) for k,v in model.state_dict().items())
            times = [r['ms'] for r in records[name][10:]]
            region = [r['region'] for r in records[name] if r['region'] is not None]
            results[name] = dict(
                metrics={'bev_ap%d'%int(t*100):eval_utils.calculate_ap(aps[name], t)[0] for t in aps[name]},
                mean_ms=float(np.mean(times)), p50_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times,95)),
                reset_frames=sum(bool(r['reset']) for r in records[name]),
                state_bytes=records[name][-1]['state_bytes'],
                region_means={k:float(np.mean([r[k] for r in region])) for k in region[0]} if region else None)
            h, w = states[name]['value'].shape[-2:]
            dense_macs, _ = same_lattice_conv_macs(model.update, h, w)
            area = results[name]['region_means']['executed_input_area_ratio'] if region else 1.
            results[name]['U_conv_compute'] = dict(
                dense_gmacs=dense_macs/1e9, executed_gmacs=dense_macs*area/1e9,
                executed_gflops=2*dense_macs*area/1e9, reduction_percent=100*(1-area),
                scope='U Conv2d only; standard 1 MAC=2 FLOPs including duplicated halos and fallback, excluding bias/BN/activation; not whole-model FLOPs or hardware instructions.')
        dense = results['dense']
        for name, result in results.items():
            result.update(ap70_delta_pp=100*(result['metrics']['bev_ap70']-dense['metrics']['bev_ap70']),
                          latency_reduction_percent=100*(1-result['mean_ms']/dense['mean_ms']))
        assert digest(source) == a.source_sha256
        if a.selector == 'learned':
            assert digest(a.selector_checkpoint) == a.selector_sha256
        if evaluation_stage == 'full':
            assert abs(dense['metrics']['bev_ap70']-source_ap) < 1e-8, 'Dense baseline replay differs from source AP'
        report = dict(status='COMPLETED', stage=evaluation_stage, scans=len(chosen),
                      labels=sum(r['supervised'] for r in chosen), checkpoint_sha256=a.source_sha256,
                      results=results, seconds=time.monotonic()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), weights_unchanged=True,
                      timing_scope='One GPU, same frames, rotating mode order; synchronized full model step, including observation, warps, selector, crop planning, U, writeback and heads; excludes loading, voxelization, H2D, postprocessing.',
                      caveats=['Pilot prefix AP is not full validation AP.',
                               'oracle_state, if enabled, executes full U before selection: diagnostic only, not deployable skipping or a formal AP upper bound.',
                               'Nominal fraction excludes mandatory coverage, age, tile expansion and dense fallback.',
                               'Executed input area counts overlapping halos; it is not a FLOP measurement.',
                               'All modes own their recursive states; skipped states may affect later refreshed states.',
                               'State uses ego-motion only; no object-motion prediction or cache-aware training.',
                               'Same-GPU paired timing is preliminary; host contention and thermal variation remain.',
                               'Peak GPU memory includes resident comparison models and teacher construction, not deployment memory.'])
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(status='COMPLETED'))
    finally:
        for hook in hooks:
            hook.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--source-sha256', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--stage', choices=['pilot','full'], default='pilot')
    p.add_argument('--frames', type=int, default=256)
    p.add_argument('--fractions', type=float, nargs='+', default=[.25,.5])
    p.add_argument('--tile', type=int, default=16)
    p.add_argument('--cache-max-age', type=float, default=.5)
    p.add_argument('--fallback-fraction', type=float, default=.85)
    p.add_argument('--execution', choices=['sparse','fixed_sparse'], default='sparse')
    p.add_argument('--selector', choices=['shallow','oracle_state','learned'], default='shallow')
    p.add_argument('--selector-checkpoint')
    p.add_argument('--selector-sha256')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = Path(a.output)
    if (a.frames < 32 or a.tile < 1 or a.batch_size < 1 or not 0<a.cache_max_age<float('inf')
            or not 0<a.fallback_fraction<=1 or not a.fractions
            or any(not 0<=f<=1 for f in a.fractions) or len(set(a.fractions)) != len(a.fractions)):
        p.error('Invalid bounded regional evaluation policy')
    if a.selector == 'learned' and (not a.selector_checkpoint or not a.selector_sha256):
        p.error('Learned selector requires a fixed checkpoint and SHA256')
    if a.worker:
        def expired(signum, frame):
            raise TimeoutError('Paired regional evaluation exceeded two hours')
        signal.signal(signal.SIGALRM, expired); signal.alarm(7200)
        try:
            worker(a)
        except Exception as exc:
            write_json(out/'status.json', dict(status='FAILED', error=repr(exc))); raise
        finally:
            signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:
        raise RuntimeError('GPU is occupied; no shared admission')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable,'-m','opencood.tools.evaluate_int_local_cache']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=process.pid, gpu=a.gpu, command=command,
        created_unix=time.time(), timeout_seconds=7200,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid, output=str(out))))


if __name__ == '__main__':
    main()

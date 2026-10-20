"""Synchronized component diagnosis; separate uninstrumented paired timing."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules import int_local_state, int_local_state_cache
from opencood.models.sub_modules.int_local_selector import selector_state
from opencood.tools.evaluate_int_local_cache import make_model
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


@contextmanager
def components(model, totals):
    """Restore all temporary wrappers, including on failure. No weight edits."""
    restored = []
    refresh_depth = [0]

    def wrap(obj, attr, name):
        original = getattr(obj, attr)
        def measured(*args, **kwargs):
            key = name
            if name == 'update':
                key = 'update_inside_refresh' if refresh_depth[0] else 'update_direct'
            torch.cuda.synchronize()
            started = time.perf_counter()
            if name == 'refresh_including_update':
                refresh_depth[0] += 1
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                totals[key] = totals.get(key, 0.) + (time.perf_counter()-started)*1000
                if name == 'refresh_including_update':
                    refresh_depth[0] -= 1
        setattr(obj, attr, measured)
        restored.append((obj, attr, original))

    wrap(model, 'spatial', 'pillar_scatter')
    wrap(model.observation, 'forward', 'observation')
    wrap(model.update, 'forward', 'update')
    for head in (model.cls_head, model.reg_head, model.dir_head):
        if head is not None:
            wrap(head, 'forward', 'heads')
    wrap(int_local_state, 'warp_bev', 'state_warp')
    if hasattr(model, 'select_request'):
        wrap(model, 'select_request', 'selector_and_request')
        wrap(int_local_state_cache, 'warp_bev', 'reference_age_warp')
        wrap(int_local_state_cache, 'refresh_tiles', 'refresh_including_update')
    try:
        yield
    finally:
        for obj, attr, original in reversed(restored):
            setattr(obj, attr, original)


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for k, v in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None,
                      periodic_last=0, max_age=.25, oracle_stage0=False, oracle_index=0,
                      support_profile=False, zero_uncovered=False).items():
        setattr(a, k, v)
    teacher, cfg, rows, provenance = setup(a)
    assert digest(a.checkpoint) == a.source_sha256
    assert digest(a.selector_checkpoint) == a.selector_sha256
    source = torch.load(a.checkpoint, map_location='cpu')
    selected = torch.load(a.selector_checkpoint, map_location='cpu')
    assert source['contract']['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert source['contract']['data_contract'] == provenance['contract']
    weights = selector_state(selected, a.source_sha256, 16, .5)
    assert selected['max_age'] == .5 and selected['fallback_fraction'] == .85
    common = dict(selector='learned', selector_weights=weights, fraction=.5,
                  tile=16, max_age=.5, fallback_fraction=.85, batch_size=32)
    policies = dict(dense={}, merged=dict(common, execution='sparse'),
                    fixed=dict(common, execution='fixed_sparse'))
    # Each instrumented/uninstrumented mode owns its complete recursive state.
    models = {name:[make_model(teacher, cfg, source, **policy) for _ in range(2)]
              for name, policy in policies.items()}
    def reject_teacher(module, inputs):
        raise RuntimeError('Teacher execution forbidden during profiling')
    hooks = [m.register_forward_pre_hook(reject_teacher) for m in
             (teacher.pillar_vfe, teacher.backbone, teacher.cls_head, teacher.reg_head)]
    write_json(out/'provenance.json', dict(provenance, checkpoint=a.checkpoint,
        source_sha256=a.source_sha256, selector_checkpoint=a.selector_checkpoint,
        selector_sha256=a.selector_sha256, frames=a.frames, fraction=.5, tile=16))
    states = {name:[None, None] for name in models}
    results = {name:[] for name in models}
    max_error = {name:0. for name in models}
    chosen = rows[:a.frames]
    _, stream = loader(chosen, cfg, 4, 303)
    try:
        with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as log:
            for index, sample in enumerate(stream):
                ego = to_device(sample['ego'], torch.device('cuda'))
                names = list(models)
                names = names[index % 3:]+names[:index % 3]
                for name in names:
                    outputs = [None, None]
                    record = dict(index=index, frame=sample['meta']['frame'], mode=name)
                    for which in ([0, 1] if index % 2 == 0 else [1, 0]):
                        model = models[name][which]
                        totals = {}
                        if which:
                            with components(model, totals):
                                torch.cuda.synchronize(); tick = time.perf_counter()
                                outputs[which], states[name][which], info = model.step(
                                    ego, states[name][which], sample['meta'])
                                torch.cuda.synchronize()
                                record['instrumented_ms'] = (time.perf_counter()-tick)*1000
                            totals['refresh_excluding_update'] = (
                                totals.get('refresh_including_update', 0.)-
                                totals.get('update_inside_refresh', 0.))
                            accounted = sum(v for k,v in totals.items()
                                            if k not in ('update_inside_refresh', 'refresh_excluding_update'))
                            totals['other_and_instrumentation'] = record['instrumented_ms']-accounted
                            record['components_ms'] = totals
                        else:
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            outputs[which], states[name][which], info = model.step(
                                ego, states[name][which], sample['meta'])
                            torch.cuda.synchronize()
                            record['uninstrumented_ms'] = (time.perf_counter()-tick)*1000
                            record['region'] = info.get('region_cache')
                            record['reset'] = info['reset']
                    error = max(float((outputs[0][k]-outputs[1][k]).abs().max()) for k in outputs[0])
                    assert all(torch.allclose(outputs[0][k], outputs[1][k], atol=1e-4, rtol=1e-4)
                               for k in outputs[0]), (name, index, error)
                    max_error[name] = max(max_error[name], error)
                    results[name].append(record)
                    log.write(json.dumps(record)+'\n')
                assert torch.allclose(states['merged'][0]['value'], states['fixed'][0]['value'],
                                      atol=1e-4, rtol=1e-4), 'Execution modes changed the policy trajectory'
                if (index+1) % 100 == 0:
                    write_json(out/'status.json', dict(status='RUNNING', scans=index+1, total=a.frames))
        report = {}
        for name, records in results.items():
            values = records[10:]
            keys = set().union(*(r['components_ms'] for r in values))
            report[name] = dict(
                uninstrumented_mean_ms=float(np.mean([r['uninstrumented_ms'] for r in values])),
                instrumented_mean_ms=float(np.mean([r['instrumented_ms'] for r in values])),
                components_mean_ms={k:float(np.mean([r['components_ms'].get(k,0.) for r in values])) for k in keys},
                prediction_max_error=max_error[name],
                partial_frames=sum(0<r['region']['actual_write_fraction']<1 for r in records if r['region']),
                resets=sum(bool(r['reset']) for r in records))
            expected = dict(source['model'])
            if name != 'dense':
                expected.update({'selector_net.'+k:v for k,v in weights.items()})
            for model in models[name]:
                assert set(expected) == set(model.state_dict())
                assert all(torch.equal(v.cpu(), expected[k]) for k,v in model.state_dict().items())
        write_json(out/'report.json', dict(status='COMPLETED', scans=len(chosen), results=report,
            weights_unchanged=True, no_ap_claim=True,
            caveats=['Validation prefix diagnostic, not full-stream AP or deployment latency.',
                     'Component timings synchronize each boundary and perturb scheduling; use uninstrumented totals separately.',
                     'refresh_including_update contains update_inside_refresh; do not sum them.',
                     'Other includes concatenation, statistics, cache writeback and instrumentation overhead.',
                     'Uninstrumented pairs still run among diagnostic copies; host/thermal contention remains.']))
        write_json(out/'status.json', dict(status='COMPLETED'))
    finally:
        for hook in hooks:
            hook.remove()


def main():
    p = argparse.ArgumentParser()
    for name in ('root','output','checkpoint','source-sha256','selector-checkpoint','selector-sha256'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--frames', type=int, default=512)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args(); out = Path(a.output)
    if not 32 <= a.frames <= 1024:
        p.error('Profiling must be bounded to 32-1024 frames')
    if a.worker:
        def expired(signum, frame):
            raise TimeoutError('Profiling exceeded 30 minutes')
        signal.signal(signal.SIGALRM, expired); signal.alarm(1800)
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
        raise RuntimeError('GPU occupied')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'opencood.tools.profile_int_local_regions']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=process.pid, command=command, gpu=a.gpu,
        created_unix=time.time(), timeout_seconds=1800,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid, output=str(out))))


if __name__ == '__main__':
    main()

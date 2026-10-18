"""Frozen 2x2 early-gate experiment, training-only calibration, audited execution."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_early_cache import EarlyCacheDetector, VERSION, refresh_reason
from opencood.tools.run_int_cache_repair import setup, evaluate
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest

COMBINATIONS = [('gru', 'stages'), ('gru', 'decoded'), ('stage1', 'stages'), ('stage1', 'decoded')]


def replay(records, threshold):
    anchor = None
    reasons = []
    for i, row in enumerate(records):
        fresh = anchor is None or bool(row['reset'])
        reason = refresh_reason(fresh, 0 if fresh else i-anchor, row['candidate_age'],
                                row['candidate_valid'], row['early_score'], threshold)
        reasons.append(reason or 'reuse')
        if reason is not None:
            anchor = i
    return dict(reuse_fraction=reasons.count('reuse')/len(reasons), counts=dict(Counter(reasons)), reasons=reasons)


def engineering(model, cfg, rows, out):
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'opencood/test',
                             '-p', 'test_int*.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError('INT tests failed')
    wrappers = [EarlyCacheDetector(model, cfg, g, c, period=1) for g, c in COMBINATIONS]
    reuse_wrappers = [EarlyCacheDetector(model, cfg, g, c, threshold=10.) for g, c in COMBINATIONS]
    states = [None]*8
    reference = None
    errors = [[] for _ in wrappers]
    counts = [0, 0, 0, 0]
    def hit(index):
        def callback(module, inputs, output):
            counts[index] += 1
        return callback
    hooks = [getattr(model.backbone.resnet, 'layer%d'%i).register_forward_hook(hit(i)) for i in range(3)]
    assert model.shrink_flag
    hooks.append(model.shrink_conv.register_forward_hook(hit(3)))
    reused = [0]*4
    _, stream = loader(rows[:8], cfg, 2, 303)
    try:
        with torch.no_grad():
            for sample in stream:
                ego = to_device(sample['ego'], torch.device('cuda'))
                expected, reference, _ = model.step(ego, reference, sample['meta'])
                for i, wrapper in enumerate(wrappers + reuse_wrappers):
                    before = list(counts)
                    pred, states[i], info = wrapper.step(ego, states[i], sample['meta'])
                    calls = [a-b for a, b in zip(counts, before)]
                    declared = [int(s['executed_input_area_ratio']) for s in info['cache_stages']] + [info['decoder_called']]
                    assert calls == declared, (calls, declared)
                    assert all(torch.isfinite(v).all() for v in pred.values())
                    if i < 4:
                        error = max(float((expected[k]-pred[k]).abs().max()) for k in expected)
                        assert error <= 1e-5, error
                        errors[i].append(error)
                    else:
                        reused[i-4] += int(info['early_refresh_reason'] is None)
    finally:
        for hook in hooks:
            hook.remove()
    return dict(status='PASSED',version=VERSION,full_refresh_errors=errors,
                real_reuse_counts=reused,actual_module_calls_match=True,
                tests='All INT tests, eight real frames, four full-refresh prediction parity checks, actual stage/shrink hooks')


def calibrate(model, cfg, rows, out, gate, kind, provenance, frames):
    wrapper = EarlyCacheDetector(model, cfg, gate, kind, period=1, probe=True)
    _, stream = loader(rows[:frames], cfg, 4, 303)
    records = []
    state = None
    with torch.no_grad(), (out/'frames.jsonl').open('x', buffering=1) as log:
        for row, sample in zip(rows[:frames], stream):
            assert row['frame'] == sample['meta']['frame']
            ego = to_device(sample['ego'], torch.device('cuda'))
            _, state, info = wrapper.step(ego, state, sample['meta'])
            record = dict(frame=row['frame'], segment=sample['meta']['segment'], **info)
            records.append(record)
            log.write(json.dumps(record)+'\n')
    assert len(records) == min(frames, len(rows))
    all_scores = [r['early_score'] for r in records if r['early_score'] is not None]
    assert all_scores and np.isfinite(all_scores).all(), 'No finite temporal comparisons in training prefix'
    scores = [r['early_score'] for r in records if r['early_score'] is not None and r['candidate_valid']]
    # Zero eligible pairs is a measured geometry limit, not a broken experiment.
    thresholds = sorted(set([0., max(all_scores)] +
        (np.quantile(scores, np.linspace(0, 1, 101)).tolist() if scores else [])))
    candidates = []
    for threshold in thresholds:
        result = replay(records, threshold)
        result.pop('reasons')
        candidates.append(dict(threshold=threshold, **result))
    selected = min(candidates, key=lambda r: (abs(r['reuse_fraction']-.4), r['threshold']))
    write_json(out/'candidates.json', candidates)
    return dict(status='COMPLETED', version=VERSION, split='train', gate=gate, cache_kind=kind,
                scans=len(records), eligible_pairs=len(scores), period=2, max_age=.25,
                max_achievable_reuse=replay(records,max(all_scores))['reuse_fraction'],
                target_reuse_fraction=.4, selected=selected,
                checkpoint_sha256=provenance['checkpoint']['sha256'], contract=provenance['contract'],
                trace_sha256=digest(out/'frames.jsonl'),
                selection='Closest training reuse budget 40%, ties lower threshold; no labels or target features used')


def check_training(model, cfg, rows, out, policy, calibration_dir, frames):
    expected = [json.loads(s) for s in (calibration_dir/'frames.jsonl').read_text().splitlines()]
    threshold = policy['selected']['threshold']
    reasons = replay(expected, threshold)['reasons']
    wrapper = EarlyCacheDetector(model, cfg, policy['gate'], policy['cache_kind'], threshold)
    _, stream = loader(rows[:frames], cfg, 4, 303)
    state = None
    counts = [0, 0, 0]
    def hit(index):
        def callback(module, inputs, output):
            counts[index] += 1
        return callback
    hooks = [getattr(model.backbone.resnet, 'layer%d'%i).register_forward_hook(hit(i)) for i in range(3)]
    checked = 0
    try:
        with torch.no_grad():
            for i, sample in enumerate(stream):
                before = list(counts)
                _, state, info = wrapper.step(to_device(sample['ego'], torch.device('cuda')), state, sample['meta'])
                assert sample['meta']['frame'] == expected[i]['frame']
                actual = info['early_refresh_reason'] or 'reuse'
                assert actual == reasons[i], (i, actual, reasons[i])
                calls = [a-b for a, b in zip(counts, before)]
                assert calls == [int(s['executed_input_area_ratio']) for s in info['cache_stages']]
                if info['early_score'] is not None:
                    assert abs(info['early_score']-expected[i]['early_score']) <= 1e-6
                checked += 1
    finally:
        for hook in hooks:
            hook.remove()
    assert checked == len(expected)
    result = dict(status='PASSED',frames=checked,actual_stage_calls=counts,
                  decisions_and_scores_match=True,calibration_sha256=digest(calibration_dir/'report.json'))
    write_json(out/'training_check.json', result)
    return result


def enrich(report, out, gate, kind, policy):
    records = [json.loads(s) for s in (out/'frames.jsonl').read_text().splitlines()]
    reused = [r for r in records if r['early_refresh_reason'] is None]
    report.update(version=VERSION, gate=gate, cache_kind=kind,
        calibration=policy, refresh_counts=dict(Counter(r['early_refresh_reason'] or 'reuse' for r in records)),
        reused_labeled_frames=sum(r['supervised'] for r in reused),
        decoder_call_fraction=float(np.mean([r['decoder_called'] for r in records])),
        backbone_conv_mac_saved_fraction=1-sum(s['executed_input_area_ratio']*m for s, m in
            zip(report['stage_means'], [20.64384,28.0756224,35.5074048]))/84.2268672,
        caveats=['Fixed frozen ANN GRU; no independent SNN benefit claimed.',
                 'GRU hidden update uses its existing aligned state; stage1 compares aligned same-layer descriptors.',
                 'One reuse frame maximum; hidden-state comparison is not valid for older deep cache anchors.',
                 'Strict coverage; all warp/selection/cache work counted. Timing excludes IO/voxelization/postprocess.',
                 'Stages cache uses current stage1 when it has already been computed; decoded cache skips the decoder.',
                 'Prefix validation and one sequential timing run do not establish stable full-validation accuracy or speed.'])
    return report


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING',phase='setup'))
    # Keep the existing verified dataset/checkpoint setup and AP implementation.
    for key, value in dict(content_calibration=None,periodic_last=0,max_age=.25,
        content_check=None,fraction=.5,tile=8,dense=False,occupancy_policy='cell',
        oracle_stage0=False,support_profile=False,oracle_index=0,zero_uncovered=False).items():
        setattr(a,key,value)
    requested = a.stage
    a.stage = 'calibrate' if requested == 'local-profile' else 'pilot'
    model,cfg,val,provenance = setup(a)
    write_json(out/'provenance.json',provenance)
    if requested == 'decoded-diagnostic':
        from opencood.tools.diagnose_int_decoded_repair import diagnose
        write_json(out/'report.json',diagnose(a,model,cfg,val,out,provenance))
    elif requested == 'learned-eval':
        from opencood.tools.train_int_light_repair import evaluate_saved
        source=Path(a.source_suite)
        original=json.loads((source/'provenance.json').read_text())
        assert original['checkpoint']['sha256']==provenance['checkpoint']['sha256']
        assert original['contract']==provenance['contract']
        policy=json.loads((source/'frozen_policies.json').read_text())['stage1_stages']
        assert policy['version']==VERSION and policy['split']=='train'
        write_json(out/'report.json',evaluate_saved(a,model,cfg,val,out,policy,provenance))
    elif requested == 'learned-repair':
        from opencood.tools.train_int_light_repair import train
        source=Path(a.source_suite)
        original=json.loads((source/'provenance.json').read_text())
        assert original['checkpoint']['sha256']==provenance['checkpoint']['sha256']
        assert original['contract']==provenance['contract']
        policy=json.loads((source/'frozen_policies.json').read_text())['stage1_stages']
        assert policy['version']==VERSION and policy['split']=='train'
        runtime=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
        train_rows=json.loads(Path(runtime['manifest']).read_text())['train']
        report=train(a,model,cfg,train_rows,val,out,policy,provenance)
        write_json(out/'report.json',report)
    elif requested == 'local-suite':
        from opencood.tools.int_local_cache_eval import run_local_suite
        source=Path(a.source_suite)
        original=json.loads((source/'provenance.json').read_text())
        assert original['checkpoint']['sha256']==provenance['checkpoint']['sha256']
        assert original['contract']==provenance['contract']
        policy=json.loads((source/'frozen_policies.json').read_text())['stage1_stages']
        assert policy['version']==VERSION and policy['split']=='train'
        report=run_local_suite(a,model,cfg,val,out,policy)
        write_json(out/'report.json',report)
    elif requested == 'local-profile':
        from opencood.tools.int_local_cache_profile import profile
        source=Path(a.source_suite)
        original=json.loads((source/'provenance.json').read_text())
        assert original['checkpoint']['sha256']==provenance['checkpoint']['sha256']
        assert original['contract']==provenance['contract']
        policy=json.loads((source/'frozen_policies.json').read_text())['stage1_stages']
        assert policy['version']==VERSION and policy['split']=='train'
        report=profile(a,model,cfg,val,out,policy)
        write_json(out/'report.json',report)
    elif requested == 'suite':
        eng = out/'engineering';eng.mkdir()
        write_json(eng/'report.json',engineering(model,cfg,val,eng))
        write_json(eng/'provenance.json',provenance)
        source = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
        train = json.loads(Path(source['manifest']).read_text())['train']
        policies = {}
        for gate,kind in COMBINATIONS:
            name = gate+'_'+kind
            write_json(out/'status.json',dict(status='RUNNING',phase='train_'+name))
            target = out/('train_'+name);target.mkdir()
            policy = calibrate(model,cfg,train,target,gate,kind,provenance,a.train_frames)
            write_json(target/'report.json',policy)
            check_training(model,cfg,train,target,policy,target,a.train_frames)
            policies[name] = policy
        # Freeze ALL four train-only choices before observing any validation AP.
        write_json(out/'frozen_policies.json', policies)
        a.engineering = str(eng/'report.json')
        a.stage = 'pilot'
        results = {}
        for gate,kind in [('gru','stages')] + COMBINATIONS:
            control = not results
            name = 'dense_control' if control else gate+'_'+kind
            policy = None if control else policies[name]
            write_json(out/'status.json',dict(status='RUNNING',phase='pilot_'+name))
            target = out/('pilot_'+name);target.mkdir()
            wrapper = EarlyCacheDetector(model,cfg,gate,kind,0. if control else policy['selected']['threshold'],
                                         period=1 if control else 2)
            torch.cuda.reset_peak_memory_stats()
            report = evaluate(a,model,cfg,val,target,wrapper)
            results[name] = enrich(report,target,gate,kind,policy)
            write_json(target/'report.json',results[name])
        write_json(out/'report.json',dict(status='COMPLETED',version=VERSION,results=results))
    elif requested == 'full':
        source = Path(a.source_suite)
        original = json.loads((source/'provenance.json').read_text())
        assert original['checkpoint']['sha256'] == provenance['checkpoint']['sha256']
        assert original['contract'] == provenance['contract']
        gate = json.loads((source/'engineering'/'report.json').read_text())
        assert gate['status'] == 'PASSED' and gate['version'] == VERSION
        name = a.gate+'_'+a.cache_kind
        policy = json.loads((source/'frozen_policies.json').read_text())[name]
        check = json.loads((source/('train_'+name)/'training_check.json').read_text())
        assert check['status'] == 'PASSED'
        assert check['calibration_sha256'] == digest(source/('train_'+name)/'report.json')
        assert policy == json.loads((source/('train_'+name)/'report.json').read_text())
        assert policy['version'] == VERSION and policy['split'] == 'train'
        a.stage = 'full'
        a.engineering = str(source/'engineering'/'report.json')
        wrapper = EarlyCacheDetector(model,cfg,a.gate,a.cache_kind,policy['selected']['threshold'])
        report = enrich(evaluate(a,model,cfg,val,out,wrapper),out,a.gate,a.cache_kind,policy)
        report['caveats'][-1] = 'One sequential measurement does not establish stable latency across repeated runs.'
        write_json(out/'report.json',report)
    else:
        raise ValueError(requested)
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--stage',choices=['suite','full','local-profile','local-suite','learned-repair','learned-eval','decoded-diagnostic'],default='suite')
    p.add_argument('--source-suite');p.add_argument('--gate',choices=['gru','stage1'],default='gru')
    p.add_argument('--cache-kind',choices=['stages','decoded'],default='stages')
    p.add_argument('--mode',choices=['gru'],default='gru')
    p.add_argument('--repair-policy',choices=['local','dense'],default='local')
    p.add_argument('--repair-target',choices=['stages','decoded'],default='stages')
    p.add_argument('--repair-checkpoint');p.add_argument('--repair-period',type=int,choices=[1,2,3],default=2)
    p.add_argument('--repair-ablation',choices=['none','zero-residual','original'],default='none')
    p.add_argument('--profile-stages',action='store_true',help='Diagnostic prefix with stage spans; learned-eval only')
    p.add_argument('--train-frames',type=int,default=1024);p.add_argument('--frames',type=int,default=256)
    p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.stage in ('full','local-profile','local-suite','learned-repair','learned-eval') and not a.source_suite:
        p.error('Full evaluation requires the original suite with frozen policies and passed checks')
    if a.stage in ('learned-eval','decoded-diagnostic') and not a.repair_checkpoint:
        p.error('learned-eval requires a completed training checkpoint')
    if a.profile_stages and a.stage!='learned-eval':p.error('Stage profiling is evaluation-only')
    if a.stage=='learned-repair' and (a.repair_period==1 or (a.repair_period==3 and a.repair_policy!='dense')):
        p.error('Training supports period2 local/dense or period3 dense')
    if a.stage=='learned-repair' and a.repair_ablation!='none':
        p.error('Ablations are evaluation-only')
    if a.repair_target=='decoded' and a.repair_policy!='dense':p.error('Decoded repair is dense-only')
    if a.worker:
        try:
            worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
            raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<12000 or int(util)>5:
        raise ValueError('GPU busy')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m','opencood.tools.run_int_early_cache']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=proc.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':
    main()

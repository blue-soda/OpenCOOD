"""Fixed-checkpoint fit and history diagnostics for the local-state baseline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.tools.run_int_cache_repair import setup, evaluate
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out = Path(a.output)
    write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    for key, value in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None,
                           periodic_last=0, max_age=.25, oracle_stage0=False, oracle_index=0,
                           support_profile=False, zero_uncovered=False).items():
        setattr(a, key, value)
    teacher, cfg, val, provenance = setup(a)
    source = Path(a.checkpoint)
    source_sha = digest(source)
    saved = torch.load(str(source), map_location='cpu')
    contract = saved['contract']
    assert contract['version'] == LocalStateDetector.version
    assert contract['phase'] in ('a', 'b')
    assert contract['teacher_sha256'] == provenance['checkpoint']['sha256']
    assert contract['data_contract'] == provenance['contract']
    trained_policy = contract['memory_policy']
    assert trained_policy == ('reset' if contract['phase'] == 'a' else 'recurrent')
    policy = trained_policy if a.memory_policy == 'trained' else 'reset'
    student = LocalStateDetector(teacher, cfg, policy).cuda().eval()
    student.load_state_dict(saved['model'], strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    runtime = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    rows = json.loads(Path(runtime['manifest']).read_text())[a.split] if a.split == 'train' else val
    assert len(rows) == contract[a.split+'_scans']
    assert sum(row['supervised'] for row in rows) == contract[a.split+'_labels']
    provenance.update(split=a.split, evaluated_checkpoint=str(source),
                      evaluated_checkpoint_sha256=source_sha, trained_memory_policy=trained_policy,
                      evaluated_memory_policy=policy, source_epoch=saved['epoch'])
    write_json(out/'provenance.json', provenance)

    # The source teacher is loaded only for config/module construction. Any
    # accidental teacher execution during either probe or full evaluation fails.
    def reject_teacher(module, inputs):
        raise RuntimeError('Teacher execution is forbidden during evaluation')
    hooks = [module.register_forward_pre_hook(reject_teacher) for module in
             (teacher.pillar_vfe, teacher.backbone, teacher.cls_head, teacher.reg_head)]
    try:
        _, stream = loader(rows[:8], cfg, 2, 303)
        state = None
        probe = []
        with torch.no_grad():
            for sample in stream:
                pred, state, info = student.step(to_device(sample['ego'], torch.device('cuda')),
                                                 state, sample['meta'])
                assert all(torch.isfinite(value).all() for value in pred.values())
                assert pred['psm'].shape[-2:] == sample['ego']['anchor_box'].shape[:2]
                assert not state['value'].requires_grad
                probe.append(dict(frame=sample['meta']['frame'], reset=info['reset']))
        if policy == 'reset':
            assert all(frame['reset'] for frame in probe)
        a.engineering = str(out/'engineering.json')
        write_json(a.engineering, dict(status='PASSED', frames=probe,
                   checkpoint_sha256=source_sha, memory_policy=policy,
                   teacher_execution_forbidden=True, optimizer_created=False))
        a.stage = 'full'
        torch.cuda.reset_peak_memory_stats()
        write_json(out/'status.json', dict(status='RUNNING', phase='full_'+a.split, scans=len(rows)))
        report = evaluate(a, teacher, cfg, rows, out, student)
        assert digest(source) == source_sha, 'Source checkpoint changed during evaluation'
        assert all(torch.equal(value.detach().cpu(), saved['model'][key])
                   for key, value in student.state_dict().items()), 'Evaluation changed weights or BN'
        report.update(scope='Fixed-weight local-state '+a.split+' diagnostic',
                      checkpoint=str(source), checkpoint_sha256=source_sha, epoch=saved['epoch'],
                      evaluation_split=a.split, trained_memory_policy=trained_policy,
                      memory_policy=policy, source_phase=contract['phase'],
                      weights_unchanged=True, optimizer_created=False,
                      timing_scope='Synchronized student model only, including alignment; excludes IO, voxelization and postprocess',
                      caveats=['No region skipping or cache speedup is tested.',
                               'Train AP is a fit diagnostic, not validation or a model selection score.',
                               'Reset is a fixed-weight inference intervention; not a separately retrained single-frame baseline.',
                               'Concurrent GPU timings are not a controlled speed comparison.'])
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(status='COMPLETED'))
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--memory-policy', choices=['trained', 'reset'], default='trained')
    parser.add_argument('--worker', action='store_true')
    a = parser.parse_args()
    out = Path(a.output)
    if a.worker:
        def expired(signum, frame):
            raise TimeoutError('Fixed-weight evaluation exceeded two-hour budget')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(7200)
        try:
            worker(a)
        except Exception as exc:
            write_json(out/'status.json', dict(status='FAILED', error=repr(exc)))
            raise
        finally:
            signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free) < 22000 or int(util) > 5:
        raise RuntimeError('Requested GPU is occupied')
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'opencood.tools.evaluate_int_local_state'] + sys.argv[1:] + ['--worker']
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (out/'run.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=process.pid, gpu=a.gpu, command=command,
        created_unix=time.time(), timeout_seconds=7200,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    print(json.dumps(dict(pid=process.pid, output=str(out))))


if __name__ == '__main__':
    main()

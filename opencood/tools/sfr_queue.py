"""Finite SFR train/eval queue. Scheduler reads status; failures stop, no automatic retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=['cv', 'none', 'motion'], required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root, arm = Path(args.root).resolve(), args.arm
    queue = root/('queue_'+arm)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=os.getcwd(),
               NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if not args.run:
        queue.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', __file__]+sys.argv[1:]+['--run']
        with (queue/'runner.log').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=env, start_new_session=True)
        metadata = dict(pid=process.pid, command=command, cwd=os.getcwd(), gpu=args.gpu, started_at=time.time(),
                        note='Finite queue, no heartbeat installed. Any failed command stops the queue; no retry or unrelated process termination.',
                        stall_advisory_seconds=1800)
        (queue/'launch.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata))
        return
    training = root/(arm+'_train_v1')
    stage, motion = ('motion', 'learned') if arm == 'motion' else ('fusion', arm)
    common = [sys.executable, '-u', 'opencood/tools/sfr_run.py', '--config', args.config, '--stage', stage, '--workers', '4']
    commands = [common+['--output', str(training), '--temporal_mode', motion, '--epochs', str(args.epochs),
                        '--split_file', str(root/'splits/fit.json')]]
    for epoch in range(1, args.epochs+1):
        checkpoint = training/('sfr_%s_epoch%d.pth' % (stage, epoch))
        modes = ['cv', 'learned'] if stage == 'motion' else [arm]
        for p in ('0.3', '0.5'):
            for mode in modes:
                output = root/('%s_dev_epoch%d_p%s_%s_seed303' % (arm, epoch, p.replace('.', ''), mode))
                commands.append(common+['--mode', 'eval', '--output', str(output), '--checkpoint', str(checkpoint),
                                        '--temporal_mode', mode, '--p', p, '--seed', '303', '--split_file', str(root/'splits/dev.json')])
    (queue/'commands.json').write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        log = queue/('command_%02d.log' % index)
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            status = dict(index=index, count=len(commands), pid=process.pid, command=command, started_at=time.time(), state='running', log=str(log))
            (queue/'status.json').write_text(json.dumps(status, indent=2))
            result = process.wait()
        status.update(exit_code=result, finished_at=time.time(), state='finished' if result == 0 else 'failed')
        (queue/('result_%02d.json' % index)).write_text(json.dumps(status, indent=2))
        (queue/'status.json').write_text(json.dumps(status, indent=2))
        if result:
            sys.exit(result)
    (queue/'completed.json').write_text(json.dumps(dict(finished_at=time.time(), commands=len(commands))))


if __name__ == '__main__':
    main()

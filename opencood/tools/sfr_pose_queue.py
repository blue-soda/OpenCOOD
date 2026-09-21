"""Finite background cache and independent pose training queue; stops on any failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    root = Path(args.root).resolve(); queue = root/'queue_pose'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=os.getcwd(),
               NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if not args.run:
        queue.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', __file__]+sys.argv[1:]+['--run']
        with (queue/'runner.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=env, start_new_session=True)
        launch = dict(pid=process.pid, command=command, cwd=os.getcwd(), gpu=args.gpu, started_at=time.time())
        (queue/'launch.json').write_text(json.dumps(launch, indent=2)); print(json.dumps(launch)); return
    common = [sys.executable, '-u', 'opencood/tools/sfr_pose_run.py', '--config', args.config]
    fit, dev = root/'pose_cache_fit_forward', root/'pose_cache_dev_forward'
    commands = [common+['--mode','cache','--training_data','--split_file',str(root/'splits/fit.json'),'--output',str(fit)],
                common+['--mode','cache','--split_file',str(root/'splits/dev.json'),'--output',str(dev)],
                common+['--mode','train','--cache',str(fit/'background_cache.pth'),'--dev_cache',str(dev/'background_cache.pth'),
                        '--output',str(root/'pose_train_forward')]]
    (queue/'commands.json').write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        path = queue/('command_%02d.log'%index)
        with path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            status = dict(index=index, count=len(commands), pid=process.pid, command=command,
                          started_at=time.time(), state='running', log=str(path))
            (queue/'status.json').write_text(json.dumps(status, indent=2))
            code = process.wait()
        status.update(exit_code=code, finished_at=time.time(), state='finished' if code==0 else 'failed')
        (queue/('result_%02d.json'%index)).write_text(json.dumps(status, indent=2))
        (queue/'status.json').write_text(json.dumps(status, indent=2))
        if code: sys.exit(code)
    (queue/'completed.json').write_text(json.dumps(dict(finished_at=time.time(), commands=len(commands))))


if __name__ == '__main__': main()

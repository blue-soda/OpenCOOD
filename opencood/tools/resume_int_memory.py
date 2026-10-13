"""Resume only interrupted INT comparisons that have not met their stopping rule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from opencood.tools.launch_int_memory import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--launch-record', default='launch_full.json',
                        help='Launch record filename within root; baseline supplement uses launch_full_single_concat.json')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    launch = json.loads((root/args.launch_record).read_text())
    plans, skipped = [], []
    for job in launch['jobs']:
        directory = Path(job['output'])
        saved = torch.load(str(directory/'latest.pth'), map_location='cpu')
        contract = saved['contract']
        lr = saved['optimizer']['param_groups'][0]['lr']
        plateau = (saved['epoch'] >= contract['min_epochs'] and
                   saved['stale'] >= contract['patience'] and lr <= contract['lr']/8)
        info = dict(mode=job['mode'], completed_epoch=saved['epoch'], stale=saved['stale'],
                    learning_rate=lr, plateau_reached=plateau, best_ap70=saved['best'])
        command = list(job['command'])
        if plateau:
            skipped.append(info)
        else:
            maximum = int(command[command.index('--max-epochs')+1])
            if saved['epoch'] >= maximum:
                raise ValueError('Epoch limit reached without plateau; explicit protocol revision required')
            for flag, key in (('--config', 'config_sha256'), ('--manifest', 'manifest_sha256'),
                              ('--checkpoint', 'initialization_sha256')):
                if digest(command[command.index(flag)+1]) != contract[key]:
                    raise ValueError('Resume input changed: '+flag)
            if command[0] != sys.executable:
                raise ValueError('Use the original Python environment')
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    tokens = proc.read_bytes().split(b'\0')
                except OSError:
                    continue
                if b'opencood.tools.run_int_epochs' in tokens and str(directory).encode() in tokens:
                    raise ValueError('Training already active for '+job['mode'])
            info.update(command=command+['--resume'], output=str(directory),
                        checkpoint_sha256=digest(directory/'latest.pth'),
                        previous_result=json.loads((directory/'result.json').read_text()))
            plans.append(info)
        del saved
    report = dict(created_unix=time.time(), skipped=skipped, jobs=plans,
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
    if args.dry_run or not plans:
        print(json.dumps(report, indent=2))
        return
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits']).decode()
    available = [int(row[0]) for row in (line.split(',') for line in snapshot.strip().splitlines())
                 if int(row[1]) >= 18000 and int(row[2]) <= 5]
    if len(available) < len(plans):
        raise RuntimeError('Not enough idle GPUs to resume the comparison')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    record = root/('resume_'+stamp+'.json')
    report['gpu_snapshot'] = snapshot
    with record.open('x') as stream:
        json.dump(report, stream, indent=2)
    for item, gpu in zip(plans, available):
        log_path = root/('resume_'+item['mode']+'_'+stamp+'.log')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(Path.cwd()),
                   NUMPY_MADVISE_HUGEPAGE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with log_path.open('x') as log:
            process = subprocess.Popen(item['command'], env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        item.update(pid=process.pid, gpu=gpu, log=str(log_path))
        temporary = Path(str(record)+'.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        os.replace(str(temporary), str(record))
    print(json.dumps(dict(record=str(record), **report), indent=2))


if __name__ == '__main__':
    main()
